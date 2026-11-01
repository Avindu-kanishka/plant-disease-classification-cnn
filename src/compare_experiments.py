
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_DIR / "reports"

# Experiment report files
ORIGINAL_REPORT = REPORT_DIR / "plantdoc_original_59_classification_report.csv"
EXPERIMENT03_REPORT = REPORT_DIR / "experiment03_59_classification_report.csv"

# Output files
SUMMARY_OUTPUT = REPORT_DIR / "experiment_comparison.csv"
CLASS_OUTPUT = REPORT_DIR / "experiment_classwise_comparison.csv"


def load_report(path):
    """Load a classification report CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")

    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str).str.strip()
    return df


def get_metric(df, row, metric):
    """Safely retrieve a metric from a classification report."""
    if row not in df.index or metric not in df.columns:
        return None

    value = pd.to_numeric(df.loc[row, metric], errors="coerce")
    return float(value) if pd.notna(value) else None


def main():
    print("\nLoading experiment reports...")

    original = load_report(ORIGINAL_REPORT)
    experiment03 = load_report(EXPERIMENT03_REPORT)

    # Overall metrics
    rows = {
        "Accuracy": ("weighted avg", "recall"),
        "Macro Precision": ("macro avg", "precision"),
        "Macro Recall": ("macro avg", "recall"),
        "Macro F1-score": ("macro avg", "f1-score"),
        "Weighted Precision": ("weighted avg", "precision"),
        "Weighted Recall": ("weighted avg", "recall"),
        "Weighted F1-score": ("weighted avg", "f1-score"),
    }

    summary = []

    for metric_name, (row, column) in rows.items():
        old_value = get_metric(original, row, column)
        new_value = get_metric(experiment03, row, column)

        summary.append({
            "Metric": metric_name,
            "Original Model": old_value,
            "Experiment 03": new_value,
            "Difference": (
                round(new_value - old_value, 4)
                if old_value is not None and new_value is not None
                else None
            )
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(SUMMARY_OUTPUT, index=False)

    # Class-wise comparison
    class_names = [
        name for name in original.index
        if name not in ["accuracy", "macro avg", "weighted avg"]
    ]

    class_results = []

    for class_name in class_names:
        if class_name not in experiment03.index:
            continue

        result = {"Class": class_name}

        for metric in ["precision", "recall", "f1-score"]:
            old_value = get_metric(original, class_name, metric)
            new_value = get_metric(experiment03, class_name, metric)

            result[f"Original {metric}"] = old_value
            result[f"Experiment 03 {metric}"] = new_value

        class_results.append(result)

    class_df = pd.DataFrame(class_results)
    class_df.to_csv(CLASS_OUTPUT, index=False)

    # Display results
    print("\n========== EXPERIMENT COMPARISON ==========\n")
    print(summary_df.to_string(index=False))

    print("\n========== CLASS-WISE COMPARISON ==========\n")
    print(class_df.to_string(index=False))

    print("\n========== BEST MODEL ==========")

    original_f1 = get_metric(original, "weighted avg", "f1-score")
    new_f1 = get_metric(experiment03, "weighted avg", "f1-score")

    if original_f1 is not None and new_f1 is not None:
        if original_f1 > new_f1:
            print("Original Model performs better (Weighted F1-score).")
        elif new_f1 > original_f1:
            print("Experiment 03 performs better (Weighted F1-score).")
        else:
            print("Both models have equal Weighted F1-score.")

    print("\nComparison completed successfully!")
    print(f"Summary: {SUMMARY_OUTPUT}")
    print(f"Class-wise results: {CLASS_OUTPUT}")


if __name__ == "__main__":
    main()
