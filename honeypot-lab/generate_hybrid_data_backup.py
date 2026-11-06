import json

import pandas as pd

import numpy as np

import os



LOG_PATH = "/home/xvhyeri/honeypot-lab/cowrie-honeypot/var/log/cowrie/cowrie.json"



def get_seed_data():

    ip_stats = {}

    if not os.path.exists(LOG_PATH):

        return []



    with open(LOG_PATH, 'r') as f:

        for line in f:

            try:

                event = json.loads(line)

                ip = event.get("src_ip")

                if not ip: continue



                if ip not in ip_stats:

                    ip_stats[ip] = {"hits": 0, "users": set(), "cmds": 0}



                ip_stats[ip]["hits"] += 1

                if event.get("username"):

                    ip_stats[ip]["users"].add(event.get("username"))



                # Check for actual command input

                if event.get("eventid") == "cowrie.command.input" or event.get("input"):

                    ip_stats[ip]["cmds"] += 1

            except:

                continue



    rows = []

    for ip, s in ip_stats.items():

        # REFINED LOGIC: Only label as Human (1) if they actually typed commands.

        # User diversity (Hydra) is no longer enough to trigger a Human label.

        label = 1 if s["cmds"] > 0 else 0

        rows.append([s["hits"], len(s["users"]), s["cmds"], label])

    return rows



# Get data from your actual logs

seed_data = get_seed_data()



# Fallback defaults if logs are empty (Bot vs Human patterns)

if not seed_data:

    print("[!] No log data found. using synthetic seeds.")

    seed_data = [[50, 1, 0, 0], [10, 1, 5, 1]]



# AUGMENTATION to 1,000 rows

augmented = []

for _ in range(1000):

    base = seed_data[np.random.randint(0, len(seed_data))]

    new_row = [

        max(1, int(base[0] + np.random.normal(0, 5))), # Hits

        max(1, int(base[1] + np.random.normal(0, 1))), # Users

        max(0, int(base[2] + np.random.normal(0, 2))), # Commands

        base[3] # Label (0 for Bot, 1 for Human)

    ]

    augmented.append(new_row)



df = pd.DataFrame(augmented, columns=['hits', 'users', 'cmds', 'is_danger'])

df.to_csv("hybrid_training_data.csv", index=False)

print("Success: 'hybrid_training_data.csv' created with 1,000 rows.")
