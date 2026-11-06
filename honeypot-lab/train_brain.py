import json

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


DATASET_PATH = "hybrid_training_data.csv"
MODEL_PATH = "mtd_brain.joblib"
METRICS_PATH = "model_metrics.json"

FEATURE_COLUMNS = [
    "hits",
    "users",
    "cmds",
]

TARGET_COLUMN = "is_danger"

# Class definition:
# 0 = Automated/Bot
# 1 = Interactive Human/High Risk


def validate_dataset(df):
    """Validate the dataset before training the model."""

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    if df[required_columns].isnull().any().any():
        raise ValueError(
            "The dataset contains missing values in the required columns."
        )

    available_classes = sorted(df[TARGET_COLUMN].unique().tolist())

    if available_classes != [0, 1]:
        raise ValueError(
            "The target column must contain both classes 0 and 1.\n"
            f"Classes found: {available_classes}"
        )

    if len(df) < 10:
        raise ValueError(
            "The dataset is too small for reliable training and testing."
        )


def main():
    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    validate_dataset(df)

    # Input features
    X = df[FEATURE_COLUMNS]

    # Target labels
    y = df[TARGET_COLUMN]

    # Split the dataset:
    # 80% for training and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Create the Random Forest classifier
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    # Train the model
    model.fit(X_train, y_train)

    # Predict using unseen testing data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    # Generate confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    # Display dataset information
    print("\n===== DATASET INFORMATION =====")
    print(f"Total records    : {len(df)}")
    print(f"Training records : {len(X_train)}")
    print(f"Testing records  : {len(X_test)}")

    print("\nClass definition:")
    print("0 = Automated/Bot")
    print("1 = Interactive Human/High Risk")

    print("\nClass distribution:")
    print(y.value_counts().sort_index())

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting class distribution:")
    print(y_test.value_counts().sort_index())

    # Display model performance
    print("\n===== MODEL PERFORMANCE =====")
    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1-score  : {f1 * 100:.2f}%")

    # Display classification report
    print("\n===== CLASSIFICATION REPORT =====")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=[
                "Automated/Bot",
                "Interactive Human",
            ],
            zero_division=0,
        )
    )

    # Display confusion matrix
    print("\n===== CONFUSION MATRIX =====")
    print(cm)

    print("\nConfusion matrix layout:")
    print("[[Correct Bot, Bot Predicted as Human],")
    print(" [Human Predicted as Bot, Correct Human]]")

    print("\nConfusion matrix interpretation:")
    print(f"Correctly classified bots       : {tn}")
    print(f"Bots classified as human        : {fp}")
    print(f"Humans classified as bots       : {fn}")
    print(f"Correctly classified humans     : {tp}")

    # Display feature importance
    print("\n===== FEATURE IMPORTANCE =====")

    feature_importance = sorted(
        zip(
            FEATURE_COLUMNS,
            model.feature_importances_,
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    for feature, importance in feature_importance:
        print(f"{feature:<10}: {importance * 100:.2f}%")

    # Save trained model
    joblib.dump(
        model,
        MODEL_PATH,
    )

    # Save model performance results
    metrics = {
        "model": "RandomForestClassifier",
        "class_definition": {
            "0": "Automated/Bot",
            "1": "Interactive Human/High Risk",
        },
        "dataset": {
            "total_records": int(len(df)),
            "training_records": int(len(X_train)),
            "testing_records": int(len(X_test)),
            "class_0_records": int((y == 0).sum()),
            "class_1_records": int((y == 1).sum()),
        },
        "performance": {
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1 * 100, 2),
        },
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "feature_importance": {
            feature: round(float(importance) * 100, 2)
            for feature, importance in feature_importance
        },
        "configuration": {
            "features": FEATURE_COLUMNS,
            "target": TARGET_COLUMN,
            "test_size": 0.20,
            "random_state": 42,
            "n_estimators": 100,
            "class_weight": "balanced",
        },
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(
            metrics,
            metrics_file,
            indent=4,
        )

    print(f"\nModel saved as: {MODEL_PATH}")
    print(f"Metrics saved as: {METRICS_PATH}")


if __name__ == "__main__":
    main()
