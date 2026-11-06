import json
import os
import random
import re
import socket
import threading
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import docker
import joblib
import pandas as pd
import requests
from docker.errors import DockerException, NotFound


# ============================================================
# CONFIGURATION
# ============================================================

BASE_PATH = Path("/home/xvhyeri/honeypot-lab")
MODEL_PATH = BASE_PATH / "mtd_brain.joblib"

COWRIE_CONTAINER_NAME = "cowrie"
DIONAEA_CONTAINER_NAME = "dionaea"

COWRIE_LOG = (
    BASE_PATH
    / "cowrie-honeypot"
    / "var"
    / "log"
    / "cowrie"
    / "cowrie.json"
)

COWRIE_LOG_DIRECTORY = (
    BASE_PATH
    / "cowrie-honeypot"
    / "var"
    / "log"
    / "cowrie"
)

COWRIE_USERDB_FILE = (
    BASE_PATH
    / "cowrie-honeypot"
    / "etc"
    / "userdb.txt"
)

# Add or change these paths if your Dionaea log is elsewhere.
DIONAEA_LOG_CANDIDATES = [
    BASE_PATH / "dionaea" / "log" / "dionaea.log",
    BASE_PATH / "dionaea" / "logs" / "dionaea.log",
    BASE_PATH / "dionaea" / "var" / "log" / "dionaea.log",
    BASE_PATH / "dionaea" / "var" / "log" / "dionaea" / "dionaea.log",
]

# Container-side ports.
COWRIE_INTERNAL_SSH = "2222/tcp"
COWRIE_INTERNAL_TELNET = "2223/tcp"

DIONAEA_INTERNAL_FTP = "21/tcp"
DIONAEA_INTERNAL_SMB = "445/tcp"
DIONAEA_INTERNAL_MYSQL = "3306/tcp"

# Default host-side ports.
DEFAULT_COWRIE_SSH_PORT = 2223
DEFAULT_COWRIE_TELNET_PORT = 2224

DEFAULT_DIONAEA_FTP_PORT = 21
DEFAULT_DIONAEA_SMB_PORT = 4445
DEFAULT_DIONAEA_MYSQL_PORT = 3306

# Randomized port range.
MTD_PORT_MIN = 20000
MTD_PORT_MAX = 29999

# Cowrie thresholds.
MIN_HUMAN_CONFIDENCE = 0.60
MIN_COMMANDS_FOR_FALLBACK = 4

# Dionaea threshold.
DIONAEA_SMB_EVENT_THRESHOLD = 4
DIONAEA_EVENT_WINDOW_SECONDS = 30

# Per-attacker cooldown.
COOLDOWN_SECONDS = 30
COOLDOWN_NOTICE_INTERVAL = 5

# Container startup validation.
CONTAINER_STARTUP_TIMEOUT = 60

ELASTICSEARCH_AUDIT_URL = (
    "http://127.0.0.1:9200/"
    "honeypot-admin-actions/_doc"
)

# Classified Cowrie events are written directly to Elasticsearch.
# This avoids calling FastAPI /api/attacks/classified because that endpoint
# can trigger a second MTD rotation for the same human classification.
ELASTICSEARCH_ATTACK_URL = (
    "http://127.0.0.1:9200/"
    "honeypot-logs-events/_doc"
)

HTTP_TIMEOUT_SECONDS = 5
ENABLE_PORT_HOP = True


# ============================================================
# INITIALIZATION
# ============================================================

try:
    docker_client = docker.from_env()
    docker_client.ping()
except DockerException as error:
    raise SystemExit(
        f"[FATAL] Unable to connect to Docker: {error}"
    ) from error

if not MODEL_PATH.exists():
    raise SystemExit(f"[FATAL] ML model not found: {MODEL_PATH}")

try:
    model = joblib.load(MODEL_PATH)
except Exception as error:
    raise SystemExit(
        f"[FATAL] Failed to load ML model: {error}"
    ) from error

CURRENT_COWRIE_SSH_PORT = DEFAULT_COWRIE_SSH_PORT
CURRENT_COWRIE_TELNET_PORT = DEFAULT_COWRIE_TELNET_PORT

CURRENT_DIONAEA_FTP_PORT = DEFAULT_DIONAEA_FTP_PORT
CURRENT_DIONAEA_SMB_PORT = DEFAULT_DIONAEA_SMB_PORT
CURRENT_DIONAEA_MYSQL_PORT = DEFAULT_DIONAEA_MYSQL_PORT

MTD_LOCK = threading.Lock()
STOP_EVENT = threading.Event()

ACTION_COUNTER = 0
ACTION_COUNTER_LOCK = threading.Lock()


# ============================================================
# GENERAL HELPERS
# ============================================================

def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def log_action(
    component: str,
    message: str,
    level: str = "INFO",
) -> None:
    """Print every important action with a UTC timestamp and sequence number."""
    global ACTION_COUNTER

    with ACTION_COUNTER_LOCK:
        ACTION_COUNTER += 1
        action_number = ACTION_COUNTER

    print(
        f"[{utc_timestamp()}] "
        f"[ACTION {action_number:04d}] "
        f"[{level}] "
        f"[{component}] "
        f"{message}",
        flush=True,
    )


def get_host_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        sock.connect(("8.8.8.8", 1))
        return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def is_port_available(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        sock.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def generate_available_port(excluded_ports: set[int]) -> int:
    log_action(
        "NETWORK",
        f"Searching for a free port in {MTD_PORT_MIN}-{MTD_PORT_MAX}; "
        f"excluded={sorted(excluded_ports)}",
    )

    for attempt in range(1, 201):
        candidate = random.randint(MTD_PORT_MIN, MTD_PORT_MAX)

        if candidate in excluded_ports:
            continue

        if is_port_available(candidate):
            log_action(
                "NETWORK",
                f"Selected free port {candidate} after {attempt} attempt(s).",
            )
            return candidate

    raise RuntimeError("Unable to find an available MTD port.")


def wait_for_port(
    port: int,
    timeout: int = CONTAINER_STARTUP_TIMEOUT,
) -> bool:
    host_ip = get_host_ip()
    deadline = time.time() + timeout

    while time.time() < deadline and not STOP_EVENT.is_set():
        results: list[bool] = []

        for target_ip in {"127.0.0.1", host_ip}:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)

            try:
                results.append(
                    sock.connect_ex((target_ip, port)) == 0
                )
            finally:
                sock.close()

        if results and all(results):
            return True

        time.sleep(1)

    return False


def find_dionaea_log() -> Path | None:
    for path in DIONAEA_LOG_CANDIDATES:
        if path.exists():
            return path

    return None


# ============================================================
# DOCKER HELPERS
# ============================================================

def get_published_port(
    container: Any,
    internal_port: str,
    default_port: int,
) -> int:
    container.reload()

    bindings = (
        container.attrs
        .get("NetworkSettings", {})
        .get("Ports", {})
        .get(internal_port)
    )

    if not bindings:
        return default_port

    try:
        return int(bindings[0]["HostPort"])
    except (KeyError, TypeError, ValueError):
        return default_port


def get_attached_networks(container: Any) -> list[str]:
    container.reload()

    networks = (
        container.attrs
        .get("NetworkSettings", {})
        .get("Networks", {})
    )

    return list(networks.keys())


def get_bind_mounts(container: Any) -> dict[str, dict[str, str]]:
    """
    Preserve only explicit host bind mounts.

    Anonymous Docker volumes are excluded so that image files are not
    accidentally hidden during container recreation.
    """
    container.reload()
    volumes: dict[str, dict[str, str]] = {}

    for mount in container.attrs.get("Mounts", []):
        if mount.get("Type") != "bind":
            continue

        source = mount.get("Source")
        destination = mount.get("Destination")
        mode = mount.get("Mode") or "rw"

        if source and destination:
            volumes[source] = {
                "bind": destination,
                "mode": mode,
            }

    return volumes


def build_cowrie_volumes() -> dict[str, dict[str, str]]:
    if not COWRIE_LOG_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Cowrie log directory does not exist: "
            f"{COWRIE_LOG_DIRECTORY}"
        )

    if not COWRIE_USERDB_FILE.exists():
        raise FileNotFoundError(
            f"Cowrie userdb file does not exist: "
            f"{COWRIE_USERDB_FILE}"
        )

    return {
        str(COWRIE_LOG_DIRECTORY): {
            "bind": "/cowrie/cowrie-git/var/log/cowrie",
            "mode": "rw",
        },
        str(COWRIE_USERDB_FILE): {
            "bind": "/cowrie/cowrie-git/data/userdb.txt",
            "mode": "rw",
        },
    }


def get_restart_policy(container: Any) -> dict[str, Any]:
    policy = (
        container.attrs
        .get("HostConfig", {})
        .get("RestartPolicy")
    )

    if not policy or not policy.get("Name"):
        return {"Name": "unless-stopped"}

    return policy


def get_optional_host_settings(container: Any) -> dict[str, Any]:
    host_config = container.attrs.get("HostConfig", {})
    settings: dict[str, Any] = {}

    if host_config.get("CapAdd"):
        settings["cap_add"] = host_config["CapAdd"]

    if host_config.get("CapDrop"):
        settings["cap_drop"] = host_config["CapDrop"]

    if host_config.get("SecurityOpt"):
        settings["security_opt"] = host_config["SecurityOpt"]

    if host_config.get("Privileged"):
        settings["privileged"] = True

    if host_config.get("Dns"):
        settings["dns"] = host_config["Dns"]

    if host_config.get("ExtraHosts"):
        settings["extra_hosts"] = host_config["ExtraHosts"]

    return settings


def connect_additional_networks(
    container: Any,
    network_names: list[str],
) -> None:
    for network_name in network_names:
        try:
            docker_client.networks.get(network_name).connect(container)
        except DockerException as error:
            print(
                f"[WARNING] Could not connect {container.name} "
                f"to {network_name}: {error}"
            )


def get_container_logs(container: Any, tail: int = 80) -> str:
    try:
        return container.logs(tail=tail).decode(
            "utf-8",
            errors="replace",
        )
    except Exception as error:
        return f"Unable to retrieve container logs: {error}"


# ============================================================
# ELASTICSEARCH AUDIT
# ============================================================

def record_mtd_event(
    *,
    service: str,
    protocol: str,
    source_ip: str,
    old_port: int,
    new_port: int | None,
    status: str,
    trigger_reason: str,
    confidence: float | None = None,
    event_count: int | None = None,
    error_message: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "@timestamp": utc_timestamp(),
        "event_type": "MTD_PORT_MUTATION",
        "requested_by": "Hybrid Adaptive MTD Engine",
        "service": service,
        "protocol": protocol,
        "source_ip": source_ip,
        "old_port": old_port,
        "new_port": new_port,
        "status": status,
        "mtd_triggered": status == "SUCCESS",
        "trigger_reason": trigger_reason,
    }

    if confidence is not None:
        payload["ml_confidence"] = confidence

    if event_count is not None:
        payload["event_count"] = event_count

    if error_message:
        payload["error_message"] = error_message

    try:
        response = requests.post(
            ELASTICSEARCH_AUDIT_URL,
            json=payload,
            timeout=5,
        )
        response.raise_for_status()
        log_action("ELASTICSEARCH", "MTD event recorded successfully.")
    except requests.RequestException as error:
        log_action(
            "ELASTICSEARCH",
            f"MTD audit failed: {error}",
            "WARNING",
        )


def record_classified_attack(
    *,
    source_ip: str,
    protocol: str,
    destination_port: int,
    classification: str,
    confidence: float,
    username: str,
    command: str | None,
    session_id: str,
    event_id: str,
    mtd_triggered: bool,
) -> None:
    """
    Write ML classification data directly to Elasticsearch so FastAPI and
    Flutter can display the real confidence and command for the session.

    This deliberately does not call /api/attacks/classified, preventing a
    duplicate automatic port rotation.
    """
    normalized_classification = classification.strip().lower()
    severity = (
        "critical"
        if normalized_classification == "human" or mtd_triggered
        else "medium"
    )

    payload: dict[str, Any] = {
        "@timestamp": utc_timestamp(),
        "event_type": "CLASSIFIED_HONEYPOT_INTERACTION",
        "source_ip": source_ip,
        "src_ip": source_ip,
        "country": "Unknown",
        "protocol": protocol.lower(),
        "destination_port": destination_port,
        "honeypot": "Cowrie",
        "classification": normalized_classification,
        "threat_type": normalized_classification,
        "confidence": confidence,
        "ml_confidence": confidence,
        "severity": severity,
        "username": username or "-",
        "command": command,
        "input": command,
        "session": session_id or "-",
        "session_id": session_id or "-",
        "cowrie_eventid": event_id,
        "mtd_triggered": mtd_triggered,
    }

    try:
        log_action(
            "ELASTICSEARCH",
            f"Writing classified Cowrie event: IP={source_ip}, "
            f"session={session_id or '-'}, class={normalized_classification}, "
            f"confidence={confidence:.2%}, command={command or '-'}",
        )
        response = requests.post(
            ELASTICSEARCH_ATTACK_URL,
            json=payload,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        log_action(
            "ELASTICSEARCH",
            "Classified Cowrie event stored successfully.",
        )
    except requests.RequestException as error:
        log_action(
            "ELASTICSEARCH",
            f"Classified-event write failed: {error}",
            "WARNING",
        )


# ============================================================
# MACHINE LEARNING
# ============================================================

def get_human_probability(features: pd.DataFrame) -> float:
    if not hasattr(model, "predict_proba"):
        prediction = int(model.predict(features)[0])
        return 1.0 if prediction == 1 else 0.0

    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)

    if 1 not in classes:
        return 0.0

    return float(probabilities[classes.index(1)])


def classify_session(
    hits: int,
    unique_users: int,
    commands: int,
) -> tuple[int, float]:
    features = pd.DataFrame(
        [[hits, unique_users, commands]],
        columns=["hits", "users", "cmds"],
    )

    prediction = int(model.predict(features)[0])
    human_probability = get_human_probability(features)

    if commands == 0:
        prediction = 0
        human_probability = 0.0

    return prediction, human_probability


# ============================================================
# COWRIE MTD
# ============================================================

def trigger_cowrie_port_hop(
    source_ip: str = "unknown",
    trigger_reason: str = "unknown",
    human_confidence: float = 0.0,
    command_count: int = 0,
) -> bool:
    global CURRENT_COWRIE_SSH_PORT
    global CURRENT_COWRIE_TELNET_PORT

    old_port = CURRENT_COWRIE_SSH_PORT
    new_port: int | None = None

    if not ENABLE_PORT_HOP:
        print("[DRY RUN] Cowrie MTD trigger detected.")
        return True

    with MTD_LOCK:
        try:
            container = docker_client.containers.get(
                COWRIE_CONTAINER_NAME
            )
            container.reload()

            old_port = get_published_port(
                container,
                COWRIE_INTERNAL_SSH,
                CURRENT_COWRIE_SSH_PORT,
            )

            telnet_port = get_published_port(
                container,
                COWRIE_INTERNAL_TELNET,
                CURRENT_COWRIE_TELNET_PORT,
            )

            new_port = generate_available_port(
                {
                    old_port,
                    telnet_port,
                    CURRENT_DIONAEA_FTP_PORT,
                    CURRENT_DIONAEA_SMB_PORT,
                    CURRENT_DIONAEA_MYSQL_PORT,
                }
            )

            config = container.attrs.get("Config", {})
            image = config.get("Image") or "cowrie/cowrie"
            environment = config.get("Env") or []
            labels = config.get("Labels") or {}
            hostname = config.get("Hostname")

            volumes = build_cowrie_volumes()
            networks = get_attached_networks(container)
            primary_network = networks[0] if networks else None

            print("\n" + "=" * 72)
            print("[!!!] COWRIE MOVING TARGET DEFENSE ACTIVATED")
            print(f"[MTD] Attacker IP     : {source_ip}")
            print(f"[MTD] Trigger reason  : {trigger_reason}")
            print(f"[MTD] Human confidence: {human_confidence:.2%}")
            print(f"[MTD] Commands        : {command_count}")
            print(f"[MTD] Old SSH port    : {old_port}")
            print(f"[MTD] New SSH port    : {new_port}")
            print(f"[MTD] Telnet port     : {telnet_port}")
            print("=" * 72)

            run_arguments: dict[str, Any] = {
                "image": image,
                "name": COWRIE_CONTAINER_NAME,
                "detach": True,
                "ports": {
                    COWRIE_INTERNAL_SSH: ("0.0.0.0", new_port),
                    COWRIE_INTERNAL_TELNET: (
                        "0.0.0.0",
                        telnet_port,
                    ),
                },
                "volumes": volumes,
                "environment": environment,
                "restart_policy": get_restart_policy(container),
                "labels": labels,
            }

            if hostname:
                run_arguments["hostname"] = hostname

            if primary_network:
                run_arguments["network"] = primary_network

            run_arguments.update(
                get_optional_host_settings(container)
            )

            container.remove(force=True)
            time.sleep(2)

            new_container = docker_client.containers.run(
                **run_arguments
            )

            connect_additional_networks(
                new_container,
                networks[1:],
            )

            if not wait_for_port(new_port):
                raise RuntimeError(
                    f"Cowrie did not become ready on port "
                    f"{new_port}.\n"
                    f"{get_container_logs(new_container)}"
                )

            new_container.reload()

            if new_container.status != "running":
                raise RuntimeError(
                    f"Cowrie status is "
                    f"'{new_container.status}'.\n"
                    f"{get_container_logs(new_container)}"
                )

            CURRENT_COWRIE_SSH_PORT = new_port
            CURRENT_COWRIE_TELNET_PORT = telnet_port

            print(
                f"[SUCCESS] Cowrie SSH moved from "
                f"{get_host_ip()}:{old_port} to "
                f"{get_host_ip()}:{new_port}."
            )

            record_mtd_event(
                service="cowrie",
                protocol="SSH",
                source_ip=source_ip,
                old_port=old_port,
                new_port=new_port,
                status="SUCCESS",
                trigger_reason=trigger_reason,
                confidence=human_confidence,
                event_count=command_count,
            )

            return True

        except Exception as error:
            error_message = f"{type(error).__name__}: {error}"
            print(f"[ERROR] Cowrie MTD failed: {error_message}")

            record_mtd_event(
                service="cowrie",
                protocol="SSH",
                source_ip=source_ip,
                old_port=old_port,
                new_port=new_port,
                status="FAILED",
                trigger_reason=trigger_reason,
                confidence=human_confidence,
                event_count=command_count,
                error_message=error_message,
            )

            return False


# ============================================================
# DIONAEA MTD
# ============================================================

def trigger_dionaea_port_hop(
    source_ip: str = "unknown",
    trigger_reason: str = "unknown",
    event_count: int = 0,
) -> bool:
    """
    Changes only Dionaea's published SMB port.

    FTP and MySQL keep their existing host ports.
    """
    global CURRENT_DIONAEA_FTP_PORT
    global CURRENT_DIONAEA_SMB_PORT
    global CURRENT_DIONAEA_MYSQL_PORT

    old_port = CURRENT_DIONAEA_SMB_PORT
    new_port: int | None = None

    if not ENABLE_PORT_HOP:
        print("[DRY RUN] Dionaea MTD trigger detected.")
        return True

    with MTD_LOCK:
        try:
            container = docker_client.containers.get(
                DIONAEA_CONTAINER_NAME
            )
            container.reload()

            ftp_port = get_published_port(
                container,
                DIONAEA_INTERNAL_FTP,
                CURRENT_DIONAEA_FTP_PORT,
            )

            old_port = get_published_port(
                container,
                DIONAEA_INTERNAL_SMB,
                CURRENT_DIONAEA_SMB_PORT,
            )

            mysql_port = get_published_port(
                container,
                DIONAEA_INTERNAL_MYSQL,
                CURRENT_DIONAEA_MYSQL_PORT,
            )

            new_port = generate_available_port(
                {
                    ftp_port,
                    old_port,
                    mysql_port,
                    CURRENT_COWRIE_SSH_PORT,
                    CURRENT_COWRIE_TELNET_PORT,
                }
            )

            config = container.attrs.get("Config", {})
            image = config.get("Image")

            if not image:
                raise RuntimeError(
                    "Unable to determine Dionaea's image name."
                )

            environment = config.get("Env") or []
            labels = config.get("Labels") or {}
            hostname = config.get("Hostname")

            # Preserve only explicit bind mounts.
            volumes = get_bind_mounts(container)

            networks = get_attached_networks(container)
            primary_network = networks[0] if networks else None

            print("\n" + "=" * 72)
            print("[!!!] DIONAEA MOVING TARGET DEFENSE ACTIVATED")
            print(f"[MTD] Attacker IP    : {source_ip}")
            print(f"[MTD] Trigger reason : {trigger_reason}")
            print(f"[MTD] SMB events     : {event_count}")
            print(f"[MTD] Old SMB port   : {old_port}")
            print(f"[MTD] New SMB port   : {new_port}")
            print(f"[MTD] FTP port       : {ftp_port}")
            print(f"[MTD] MySQL port     : {mysql_port}")
            print("=" * 72)

            run_arguments: dict[str, Any] = {
                "image": image,
                "name": DIONAEA_CONTAINER_NAME,
                "detach": True,
                "ports": {
                    DIONAEA_INTERNAL_FTP: (
                        "0.0.0.0",
                        ftp_port,
                    ),
                    DIONAEA_INTERNAL_SMB: (
                        "0.0.0.0",
                        new_port,
                    ),
                    DIONAEA_INTERNAL_MYSQL: (
                        "0.0.0.0",
                        mysql_port,
                    ),
                },
                "volumes": volumes,
                "environment": environment,
                "restart_policy": get_restart_policy(container),
                "labels": labels,
            }

            if hostname:
                run_arguments["hostname"] = hostname

            if primary_network:
                run_arguments["network"] = primary_network

            run_arguments.update(
                get_optional_host_settings(container)
            )

            container.remove(force=True)
            time.sleep(2)

            new_container = docker_client.containers.run(
                **run_arguments
            )

            connect_additional_networks(
                new_container,
                networks[1:],
            )

            if not wait_for_port(new_port):
                raise RuntimeError(
                    f"Dionaea SMB did not become ready on "
                    f"port {new_port}.\n"
                    f"{get_container_logs(new_container)}"
                )

            new_container.reload()

            if new_container.status != "running":
                raise RuntimeError(
                    f"Dionaea status is "
                    f"'{new_container.status}'.\n"
                    f"{get_container_logs(new_container)}"
                )

            CURRENT_DIONAEA_FTP_PORT = ftp_port
            CURRENT_DIONAEA_SMB_PORT = new_port
            CURRENT_DIONAEA_MYSQL_PORT = mysql_port

            print(
                f"[SUCCESS] Dionaea SMB moved from "
                f"{get_host_ip()}:{old_port} to "
                f"{get_host_ip()}:{new_port}."
            )
            print(f"[SUCCESS] Dionaea FTP remains on {ftp_port}.")
            print(
                f"[SUCCESS] Dionaea MySQL remains on "
                f"{mysql_port}."
            )

            record_mtd_event(
                service="dionaea",
                protocol="SMB",
                source_ip=source_ip,
                old_port=old_port,
                new_port=new_port,
                status="SUCCESS",
                trigger_reason=trigger_reason,
                event_count=event_count,
            )

            return True

        except Exception as error:
            error_message = f"{type(error).__name__}: {error}"
            print(f"[ERROR] Dionaea MTD failed: {error_message}")

            record_mtd_event(
                service="dionaea",
                protocol="SMB",
                source_ip=source_ip,
                old_port=old_port,
                new_port=new_port,
                status="FAILED",
                trigger_reason=trigger_reason,
                event_count=event_count,
                error_message=error_message,
            )

            return False


# ============================================================
# DIONAEA LOG PARSING
# ============================================================

IP_PORT_PATTERN = re.compile(
    r"(?P<ip>(?:\d{1,3}\.){3}\d{1,3})"
    r"(?::|/)"
    r"(?P<port>\d{1,5})"
)

PLAIN_IP_PATTERN = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)


def parse_dionaea_smb_event(
    line: str,
) -> tuple[str, int] | None:
    """
    Supports common Dionaea text-log formats.

    It returns (source_ip, destination_port) for probable SMB events.
    """
    lowered = line.lower()

    if not any(
        keyword in lowered
        for keyword in (
            "connection",
            "connect",
            "accept",
            "incident",
            "smb",
            "445",
        )
    ):
        return None

    pairs = [
        (
            match.group("ip"),
            int(match.group("port")),
        )
        for match in IP_PORT_PATTERN.finditer(line)
    ]

    if len(pairs) >= 2:
        source_ip = pairs[0][0]
        destination_port = pairs[-1][1]

        if destination_port == 445 or "smb" in lowered:
            return source_ip, destination_port

    ips = PLAIN_IP_PATTERN.findall(line)

    if ips and ("smb" in lowered or "445" in lowered):
        return ips[0], 445

    return None


# ============================================================
# COWRIE MONITOR
# ============================================================

def monitor_cowrie() -> None:
    log_action("COWRIE", f"Monitoring log: {COWRIE_LOG}")

    session_data: dict[str, dict[str, Any]] = {}
    log_file: Any | None = None
    current_inode: int | None = None

    while not STOP_EVENT.is_set():
        try:
            if not COWRIE_LOG.exists():
                print(f"[COWRIE] Waiting for log: {COWRIE_LOG}")
                STOP_EVENT.wait(2)
                continue

            inode = COWRIE_LOG.stat().st_ino

            if log_file is None or inode != current_inode:
                if log_file is not None:
                    log_file.close()

                log_file = COWRIE_LOG.open(
                    "r",
                    encoding="utf-8",
                    errors="replace",
                )

                # Avoid replaying previous events.
                log_file.seek(0, os.SEEK_END)
                current_inode = inode
                log_action("COWRIE", "Log reader ready; waiting for new events.")

            line = log_file.readline()

            if not line:
                STOP_EVENT.wait(0.2)
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_id = event.get("eventid", "")

            if event_id not in {
                "cowrie.login.failed",
                "cowrie.login.success",
                "cowrie.command.input",
            }:
                continue

            source_ip = str(event.get("src_ip", "unknown"))
            current_time = time.time()
            session_id = str(
                event.get("session")
                or event.get("session_id")
                or "-"
            )
            event_username = str(event.get("username") or "-")
            event_command = (
                str(event.get("input")).strip()
                if event.get("input") is not None
                else None
            )

            attacker = session_data.setdefault(
                source_ip,
                {
                    "hits": 0,
                    "users": set(),
                    "cmds": 0,
                    "last_username": "-",
                    "last_command": None,
                    "session_id": "-",
                    "cooldown_until": 0.0,
                    "last_notice": 0.0,
                },
            )

            if current_time < attacker["cooldown_until"]:
                if (
                    current_time - attacker["last_notice"]
                    >= COOLDOWN_NOTICE_INTERVAL
                ):
                    remaining = max(
                        0,
                        int(
                            attacker["cooldown_until"]
                            - current_time
                        ),
                    )
                    print(
                        f"[COWRIE COOLDOWN] Events from "
                        f"{source_ip} cannot trigger MTD for "
                        f"{remaining}s."
                    )
                    attacker["last_notice"] = current_time

                continue

            attacker["session_id"] = session_id

            if event_username != "-":
                attacker["last_username"] = event_username

            if event_id in {
                "cowrie.login.failed",
                "cowrie.login.success",
            }:
                attacker["hits"] += 1
                attacker["users"].add(event_username)

            elif event_id == "cowrie.command.input":
                attacker["cmds"] += 1

                if event_command:
                    attacker["last_command"] = event_command

            hits = int(attacker["hits"])
            unique_users = len(attacker["users"])
            commands = int(attacker["cmds"])

            prediction, human_probability = classify_session(
                hits,
                unique_users,
                commands,
            )

            bot_probability = max(
                0.0,
                1.0 - human_probability,
            )

            print(
                f"[COWRIE ML] IP={source_ip} | "
                f"Event={event_id} | "
                f"Hits={hits} | Users={unique_users} | "
                f"Cmds={commands} | "
                f"Bot={bot_probability:.2%} | "
                f"Human={human_probability:.2%} | "
                f"Prediction={prediction}"
            )

            ml_trigger = (
                prediction == 1
                and human_probability >= MIN_HUMAN_CONFIDENCE
                and commands >= 1
            )

            behavioural_trigger = (
                hits >= 1
                and commands >= MIN_COMMANDS_FOR_FALLBACK
            )

            should_trigger = ml_trigger or behavioural_trigger
            classification = "human" if prediction == 1 else "bot"

            # Store command events and final trigger decisions so the backend
            # can merge the command and ML confidence into one session card.
            if event_id == "cowrie.command.input" or should_trigger:
                record_classified_attack(
                    source_ip=source_ip,
                    protocol="ssh",
                    destination_port=CURRENT_COWRIE_SSH_PORT,
                    classification=classification,
                    confidence=human_probability,
                    username=str(attacker["last_username"]),
                    command=(
                        event_command
                        or attacker.get("last_command")
                    ),
                    session_id=str(attacker["session_id"]),
                    event_id=event_id,
                    mtd_triggered=False,
                )

            if not should_trigger:
                continue

            trigger_reason = (
                "Random Forest human classification"
                if ml_trigger
                else "interactive command threshold"
            )

            success = trigger_cowrie_port_hop(
                source_ip=source_ip,
                trigger_reason=trigger_reason,
                human_confidence=human_probability,
                command_count=commands,
            )

            if success:
                record_classified_attack(
                    source_ip=source_ip,
                    protocol="ssh",
                    destination_port=CURRENT_COWRIE_SSH_PORT,
                    classification="human",
                    confidence=human_probability,
                    username=str(attacker["last_username"]),
                    command=attacker.get("last_command"),
                    session_id=str(attacker["session_id"]),
                    event_id=event_id,
                    mtd_triggered=True,
                )

                attacker["hits"] = 0
                attacker["users"] = set()
                attacker["cmds"] = 0
                attacker["last_username"] = "-"
                attacker["last_command"] = None
                attacker["session_id"] = "-"
                attacker["cooldown_until"] = (
                    time.time() + COOLDOWN_SECONDS
                )
                attacker["last_notice"] = 0.0

                if log_file is not None:
                    log_file.close()
                    log_file = None

        except FileNotFoundError:
            if log_file is not None:
                log_file.close()
                log_file = None

            STOP_EVENT.wait(2)

        except Exception as error:
            print(
                f"[COWRIE ERROR] "
                f"{type(error).__name__}: {error}"
            )
            STOP_EVENT.wait(2)

    if log_file is not None:
        log_file.close()


# ============================================================
# DIONAEA MONITOR
# ============================================================

def monitor_dionaea() -> None:
    session_data: dict[str, dict[str, Any]] = {}
    log_file: Any | None = None
    active_log: Path | None = None
    current_inode: int | None = None

    while not STOP_EVENT.is_set():
        try:
            detected_log = find_dionaea_log()

            if detected_log is None:
                print(
                    "[DIONAEA] Log not found. Checked:\n  - "
                    + "\n  - ".join(
                        str(path)
                        for path in DIONAEA_LOG_CANDIDATES
                    )
                )
                STOP_EVENT.wait(5)
                continue

            inode = detected_log.stat().st_ino

            if (
                log_file is None
                or detected_log != active_log
                or inode != current_inode
            ):
                if log_file is not None:
                    log_file.close()

                log_file = detected_log.open(
                    "r",
                    encoding="utf-8",
                    errors="replace",
                )

                # Avoid replaying previous Dionaea events.
                log_file.seek(0, os.SEEK_END)

                active_log = detected_log
                current_inode = inode

                log_action("DIONAEA", f"Monitoring log: {active_log}")

            line = log_file.readline()

            if not line:
                STOP_EVENT.wait(0.2)
                continue

            parsed_event = parse_dionaea_smb_event(line)

            if parsed_event is None:
                continue

            source_ip, _destination_port = parsed_event
            current_time = time.time()

            attacker = session_data.setdefault(
                source_ip,
                {
                    "events": deque(),
                    "cooldown_until": 0.0,
                    "last_notice": 0.0,
                },
            )

            if current_time < attacker["cooldown_until"]:
                if (
                    current_time - attacker["last_notice"]
                    >= COOLDOWN_NOTICE_INTERVAL
                ):
                    remaining = max(
                        0,
                        int(
                            attacker["cooldown_until"]
                            - current_time
                        ),
                    )
                    print(
                        f"[DIONAEA COOLDOWN] Events from "
                        f"{source_ip} cannot trigger MTD for "
                        f"{remaining}s."
                    )
                    attacker["last_notice"] = current_time

                continue

            event_times: deque[float] = attacker["events"]
            event_times.append(current_time)

            while (
                event_times
                and current_time - event_times[0]
                > DIONAEA_EVENT_WINDOW_SECONDS
            ):
                event_times.popleft()

            event_count = len(event_times)

            print(
                f"[DIONAEA SMB] IP={source_ip} | "
                f"Events={event_count}/"
                f"{DIONAEA_SMB_EVENT_THRESHOLD} | "
                f"Window={DIONAEA_EVENT_WINDOW_SECONDS}s"
            )

            if event_count < DIONAEA_SMB_EVENT_THRESHOLD:
                continue

            trigger_reason = (
                f"{event_count} SMB connection events within "
                f"{DIONAEA_EVENT_WINDOW_SECONDS} seconds"
            )

            success = trigger_dionaea_port_hop(
                source_ip=source_ip,
                trigger_reason=trigger_reason,
                event_count=event_count,
            )

            if success:
                event_times.clear()
                attacker["cooldown_until"] = (
                    time.time() + COOLDOWN_SECONDS
                )
                attacker["last_notice"] = 0.0

                if log_file is not None:
                    log_file.close()
                    log_file = None

        except FileNotFoundError:
            if log_file is not None:
                log_file.close()
                log_file = None

            STOP_EVENT.wait(2)

        except Exception as error:
            print(
                f"[DIONAEA ERROR] "
                f"{type(error).__name__}: {error}"
            )
            STOP_EVENT.wait(2)

    if log_file is not None:
        log_file.close()


# ============================================================
# STARTUP INFORMATION
# ============================================================

def load_current_ports() -> None:
    global CURRENT_COWRIE_SSH_PORT
    global CURRENT_COWRIE_TELNET_PORT
    global CURRENT_DIONAEA_FTP_PORT
    global CURRENT_DIONAEA_SMB_PORT
    global CURRENT_DIONAEA_MYSQL_PORT

    try:
        cowrie = docker_client.containers.get(
            COWRIE_CONTAINER_NAME
        )

        CURRENT_COWRIE_SSH_PORT = get_published_port(
            cowrie,
            COWRIE_INTERNAL_SSH,
            DEFAULT_COWRIE_SSH_PORT,
        )

        CURRENT_COWRIE_TELNET_PORT = get_published_port(
            cowrie,
            COWRIE_INTERNAL_TELNET,
            DEFAULT_COWRIE_TELNET_PORT,
        )

    except NotFound:
        log_action("STARTUP", "Cowrie container was not found.", "WARNING")

    try:
        dionaea = docker_client.containers.get(
            DIONAEA_CONTAINER_NAME
        )

        CURRENT_DIONAEA_FTP_PORT = get_published_port(
            dionaea,
            DIONAEA_INTERNAL_FTP,
            DEFAULT_DIONAEA_FTP_PORT,
        )

        CURRENT_DIONAEA_SMB_PORT = get_published_port(
            dionaea,
            DIONAEA_INTERNAL_SMB,
            DEFAULT_DIONAEA_SMB_PORT,
        )

        CURRENT_DIONAEA_MYSQL_PORT = get_published_port(
            dionaea,
            DIONAEA_INTERNAL_MYSQL,
            DEFAULT_DIONAEA_MYSQL_PORT,
        )

    except NotFound:
        log_action("STARTUP", "Dionaea container was not found.", "WARNING")


def display_configuration() -> None:
    print("=" * 72)
    print(" HYBRID ADAPTIVE HONEYPOT MTD ORCHESTRATOR")
    print("=" * 72)
    print(f"[PORT] Cowrie SSH    : {CURRENT_COWRIE_SSH_PORT}")
    print(
        f"[PORT] Cowrie Telnet : "
        f"{CURRENT_COWRIE_TELNET_PORT}"
    )
    print(f"[PORT] Dionaea FTP   : {CURRENT_DIONAEA_FTP_PORT}")
    print(f"[PORT] Dionaea SMB   : {CURRENT_DIONAEA_SMB_PORT}")
    print(
        f"[PORT] Dionaea MySQL : "
        f"{CURRENT_DIONAEA_MYSQL_PORT}"
    )
    print(
        f"[MODE] Port hopping  : "
        f"{'ENABLED' if ENABLE_PORT_HOP else 'DRY RUN'}"
    )
    print("=" * 72)


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:
    log_action("SYSTEM", "Starting Hybrid Adaptive Honeypot MTD orchestrator.")
    log_action("STARTUP", "Reading current Docker port mappings.")
    load_current_ports()
    display_configuration()

    monitor_threads = [
        threading.Thread(
            target=monitor_cowrie,
            name="cowrie-monitor",
            daemon=True,
        ),
        threading.Thread(
            target=monitor_dionaea,
            name="dionaea-monitor",
            daemon=True,
        ),
    ]

    for thread in monitor_threads:
        log_action("SYSTEM", f"Starting monitoring thread: {thread.name}")
        thread.start()

    log_action("SYSTEM", "Cowrie and Dionaea monitoring threads are running.")

    try:
        while all(
            thread.is_alive()
            for thread in monitor_threads
        ):
            time.sleep(1)

    except KeyboardInterrupt:
        log_action("SYSTEM", "Stopping MTD orchestrator...", "WARNING")
        STOP_EVENT.set()

    for thread in monitor_threads:
        thread.join(timeout=5)

    log_action("SYSTEM", "MTD orchestrator stopped.")


if __name__ == "__main__":
    main()
