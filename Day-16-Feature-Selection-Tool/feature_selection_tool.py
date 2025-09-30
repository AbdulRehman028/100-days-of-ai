import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LinearRegression, LogisticRegression


def determine_task_type(y: pd.Series, explicit_task: Optional[str] = None) -> str:
    """Infer the task type (classification or regression)."""
    if explicit_task:
        return explicit_task

    if y.dtype == object or y.dtype.name in {"category", "bool"}:
        return "classification"

    unique_values = y.nunique(dropna=True)
    if unique_values <= 10:
        return "classification"
    return "regression"


def prepare_features(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split features and target, encoding categorical variables using one-hot encoding."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    X = df.drop(columns=[target])
    y = df[target]

    X_encoded = pd.get_dummies(X, drop_first=False)

    if X_encoded.empty:
        raise ValueError("No feature columns available after preprocessing.")

    return X_encoded, y


def select_features_rfe(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int,
    task_type: str,
) -> Tuple[List[str], pd.DataFrame]:
    """Select features using Recursive Feature Elimination (RFE)."""
    if task_type == "classification":
        estimator = LogisticRegression(max_iter=1000)
    else:
        estimator = LinearRegression()

    selector = RFE(estimator=estimator, n_features_to_select=n_features)
    selector.fit(X, y)

    support_mask = selector.get_support()
    selected_features = X.columns[support_mask].tolist()

    coefficients = getattr(selector.estimator_, "coef_", None)
    if coefficients is not None:
        coefficients = np.atleast_2d(coefficients)
        scores = np.mean(np.abs(coefficients), axis=0)
    else:
        scores = np.ones(len(X.columns))

    scores_df = pd.DataFrame(
        {
            "feature": X.columns,
            "selected": support_mask,
            "ranking": selector.ranking_,
            "score": scores,
        }
    ).sort_values(by=["selected", "score"], ascending=[False, False])

    return selected_features, scores_df


def select_features_mutual_info(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int,
    task_type: str,
    random_state: int = 42,
) -> Tuple[List[str], pd.DataFrame]:
    """Select features using mutual information."""
    discrete_features = "auto"
    if task_type == "classification":
        scores = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=random_state)
    else:
        scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=random_state)

    score_series = pd.Series(scores, index=X.columns, name="score")
    top_features = score_series.nlargest(n_features)
    scores_df = score_series.sort_values(ascending=False).reset_index().rename(columns={"index": "feature"})

    return top_features.index.tolist(), scores_df


def save_reduced_dataset(
    df: pd.DataFrame,
    selected_features: List[str],
    target: str,
    output_path: Path,
) -> None:
    """Persist the reduced dataset containing selected features and the target column."""
    columns_to_save = selected_features + [target]
    reduced_df = df[columns_to_save]
    reduced_df.to_csv(output_path, index=False)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select top features from a dataset using RFE or mutual information.",
    )
    parser.add_argument("--input", required=True, help="Path to the input CSV dataset.")
    parser.add_argument("--target", required=True, help="Name of the target column.")
    parser.add_argument(
        "--method",
        choices=["rfe", "mutual_info"],
        default="mutual_info",
        help="Feature selection method to apply (default: mutual_info).",
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        help="Specify task type. If omitted, it will be inferred from the target column.",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=5,
        help="Number of features to select (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("selected_features.csv"),
        help="Path to save the reduced dataset (default: selected_features.csv).",
    )
    parser.add_argument(
        "--save-scores",
        type=Path,
        help="Optional path to save feature scores/rankings as CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file '{input_path}' does not exist.")

    df = pd.read_csv(input_path)
    X, y = prepare_features(df, args.target)

    task_type = determine_task_type(y, explicit_task=args.task)

    n_features = min(args.n_features, X.shape[1])
    if n_features < args.n_features:
        print(
            f"⚠️ Requested {args.n_features} features but only {X.shape[1]} available. Using {n_features}."
        )

    if args.method == "rfe":
        selected_features, scores_df = select_features_rfe(X, y, n_features, task_type)
    else:
        selected_features, scores_df = select_features_mutual_info(X, y, n_features, task_type)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_reduced_dataset(df, selected_features, args.target, args.output)

    if args.save_scores:
        args.save_scores.parent.mkdir(parents=True, exist_ok=True)
        scores_df.to_csv(args.save_scores, index=False)

    print("✅ Feature selection completed!")
    print(f"Task type: {task_type}")
    print(f"Method: {args.method}")
    print(f"Selected features: {', '.join(selected_features)}")
    print(f"Reduced dataset saved to: {args.output.resolve()}")
    if args.save_scores:
        print(f"Feature scores saved to: {args.save_scores.resolve()}")


if __name__ == "__main__":
    main()