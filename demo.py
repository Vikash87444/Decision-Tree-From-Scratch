"""Run the decision-tree assignment examples: `python demo.py`."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from decision_tree import DecisionTreeClassifier, entropy, information_gain


def play_tennis_dataset():
    """Classic 14-observation categorical weather example."""
    X = [
        ["Sunny", "Hot", "High", "Weak"],
        ["Sunny", "Hot", "High", "Strong"],
        ["Overcast", "Hot", "High", "Weak"],
        ["Rain", "Mild", "High", "Weak"],
        ["Rain", "Cool", "Normal", "Weak"],
        ["Rain", "Cool", "Normal", "Strong"],
        ["Overcast", "Cool", "Normal", "Strong"],
        ["Sunny", "Mild", "High", "Weak"],
        ["Sunny", "Cool", "Normal", "Weak"],
        ["Rain", "Mild", "Normal", "Weak"],
        ["Sunny", "Mild", "Normal", "Strong"],
        ["Overcast", "Mild", "High", "Strong"],
        ["Overcast", "Hot", "Normal", "Weak"],
        ["Rain", "Mild", "High", "Strong"],
    ]
    y = ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
    return X, y, ["Outlook", "Temperature", "Humidity", "Wind"]


def synthetic_numeric_dataset():
    """Generate reproducible numerical features without outside downloads."""
    rng = random.Random(42)
    X = [[round(rng.uniform(0, 10), 3), round(rng.uniform(0, 10), 3)] for _ in range(240)]
    y = ["High" if temperature > 6 and humidity < 4 else "Low" for temperature, humidity in X]
    indices = list(range(len(X)))
    rng.shuffle(indices)
    train_ids, test_ids = indices[:168], indices[168:]
    return (
        [X[i] for i in train_ids], [y[i] for i in train_ids],
        [X[i] for i in test_ids], [y[i] for i in test_ids],
    )


def main():
    parser = argparse.ArgumentParser(description="Entropy-based decision tree from scratch")
    parser.add_argument("--max-depth", type=int, default=4, help="Max tree depth (default 4, 0 = root only)")
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs", help="Directory to write DOT visualization")
    args = parser.parse_args()

    print("=== Example 1: Categorical Play Tennis dataset ===")
    weather_X, weather_y, names = play_tennis_dataset()
    weather_tree = DecisionTreeClassifier(max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf)
    weather_tree.fit(weather_X, weather_y, feature_names=names)
    print(f"Initial label entropy: {entropy(weather_y):.6f} bits")
    print(f"Training accuracy: {weather_tree.score(weather_X, weather_y):.2%}")
    print(f"Tree depth: {weather_tree.get_depth()} | Leaves: {weather_tree.get_n_leaves()}")
    print(weather_tree.export_text())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dot_file = output_dir / "play_tennis_tree.dot"
    dot_file.write_text(weather_tree.export_dot(), encoding="utf-8")
    svg_file = output_dir / "play_tennis_tree.svg"
    svg_file.write_text(weather_tree.export_svg(), encoding="utf-8")
    print(f"\nGraphviz tree file: {dot_file}")
    print(f"Directly viewable SVG tree: {svg_file}")
    print(f"To render a PNG if Graphviz is installed: dot -Tpng {dot_file} -o {output_dir / 'play_tennis_tree.png'}")

    print("\n=== Example 2: Numerical synthetic dataset ===")
    X_train, y_train, X_test, y_test = synthetic_numeric_dataset()
    numeric_tree = DecisionTreeClassifier(max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf)
    numeric_tree.fit(X_train, y_train, feature_names=["Temperature", "Humidity"])
    print(f"Train rows: {len(y_train)} | Test rows: {len(y_test)}")
    print(f"Training accuracy: {numeric_tree.score(X_train, y_train):.2%}")
    print(f"Held-out test accuracy: {numeric_tree.score(X_test, y_test):.2%}")
    print(f"Tree depth: {numeric_tree.get_depth()} | Leaves: {numeric_tree.get_n_leaves()}")
    print(numeric_tree.export_text())
    numeric_dot = output_dir / "numeric_tree.dot"
    numeric_dot.write_text(numeric_tree.export_dot(), encoding="utf-8")
    numeric_svg = output_dir / "numeric_tree.svg"
    numeric_svg.write_text(numeric_tree.export_svg(), encoding="utf-8")
    print(f"Graphviz tree file: {numeric_dot}")
    print(f"Directly viewable SVG tree: {numeric_svg}")

    # Hand-check one information gain calculation on a perfectly separable set.
    print(f"\nInformation gain, [A,A,B,B] -> [A,A] + [B,B]: {information_gain(['A', 'A', 'B', 'B'], ['A', 'A'], ['B', 'B']):.3f} bit")


if __name__ == "__main__":
    main()
