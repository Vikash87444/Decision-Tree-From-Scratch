"""A small, dependency-free decision-tree classifier built from scratch.

The splitting criterion is Shannon entropy / information gain. Continuous
features use binary <= threshold splits; categorical features use == value.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite, log2
from numbers import Real
from typing import Any, Sequence


def entropy(labels: Sequence[Any]) -> float:
    """Shannon entropy H(Y) = -sum(p(c) * log2(p(c)))."""
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = len(labels)
    # Clamp the sign of a pure group to avoid displaying negative zero.
    return max(0.0, -sum((count / total) * log2(count / total) for count in counts.values()))


def information_gain(parent: Sequence[Any], left: Sequence[Any], right: Sequence[Any]) -> float:
    """Reduction in entropy from partitioning a parent into two children."""
    if not parent or len(left) + len(right) != len(parent):
        raise ValueError("Children must partition a nonempty parent.")
    n = len(parent)
    return entropy(parent) - (len(left) / n) * entropy(left) - (len(right) / n) * entropy(right)


@dataclass
class TreeNode:
    prediction: Any
    samples: int
    counts: dict[Any, int]
    impurity: float
    depth: int
    feature_index: int | None = None
    threshold: float | None = None
    category: Any = None
    categorical: bool = False
    gain: float = 0.0
    left: TreeNode | None = None
    right: TreeNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


class DecisionTreeClassifier:
    """Greedy binary classification tree trained using information gain.

    Parameters
    ----------
    max_depth : int | None
        Maximum number of edges from root to leaf. 0 means root-only tree;
        None means no explicit depth limit.
    min_samples_split : int
        Minimum observations in a node before attempting to split it.
    min_samples_leaf : int
        Minimum observations required in *each* child of a split.
    min_information_gain : float
        Required entropy improvement to accept a split (non-negative).

    Notes
    -----
    Numeric columns support integers and finite floats; categorical columns
    support non-numeric hashable values. Missing values are not supported.
    One-vs-rest splitting is used for multi-category features.
    """

    def __init__(
        self,
        max_depth: int | None = 4,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_information_gain: float = 0.0,
    ) -> None:
        if max_depth is not None and (type(max_depth) is not int or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")
        if type(min_samples_split) is not int or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2")
        if type(min_samples_leaf) is not int or min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be an integer >= 1")
        if not isinstance(min_information_gain, Real) or not isfinite(min_information_gain) or min_information_gain < 0:
            raise ValueError("min_information_gain must be a finite non-negative number")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = float(min_information_gain)
        self.root_: TreeNode | None = None
        self.n_features_in_: int | None = None
        self.feature_names_: list[str] = []
        self.feature_is_numeric_: list[bool] = []
        self.classes_: list[Any] = []

    def fit(
        self,
        X: Sequence[Sequence[Any]],
        y: Sequence[Any],
        feature_names: Sequence[str] | None = None,
    ) -> DecisionTreeClassifier:
        """Build the tree from rows X and their class labels y."""
        rows = [list(row) for row in X]
        labels = list(y)
        if not rows or not labels or len(rows) != len(labels):
            raise ValueError("X and y must contain the same positive number of rows")
        width = len(rows[0])
        if width == 0 or any(len(row) != width for row in rows):
            raise ValueError("X must be a rectangular matrix with at least one feature")
        try:
            self.classes_ = sorted(set(labels), key=lambda c: repr(c))
        except TypeError as exc:
            raise ValueError("Class labels must be hashable") from exc
        if any(v is None for v in labels):
            raise ValueError("Missing class labels are not supported")
        if feature_names is None:
            names = [f"feature_{i}" for i in range(width)]
        else:
            names = list(feature_names)
            if len(names) != width or any(not isinstance(n, str) for n in names):
                raise ValueError("feature_names must provide one string per feature")

        numeric = []
        for column in zip(*rows):
            if any(v is None for v in column):
                raise ValueError("Missing feature values are not supported")
            is_numeric = all(isinstance(v, Real) and not isinstance(v, bool) for v in column)
            if is_numeric:
                if any(not isfinite(v) for v in column):
                    raise ValueError("Numeric features must contain only finite values")
            else:
                if any(isinstance(v, Real) and not isinstance(v, bool) for v in column):
                    raise ValueError("A column cannot mix numeric and categorical values")
                try:
                    set(column)
                except TypeError as exc:
                    raise ValueError("Categorical feature values must be hashable") from exc
            numeric.append(is_numeric)

        self.n_features_in_ = width
        self.feature_names_ = names
        self.feature_is_numeric_ = numeric
        self.root_ = None
        self._X = rows
        self._y = labels
        try:
            self.root_ = self._grow(list(range(len(rows))), depth=0)
        finally:
            # Training examples do not need to remain in the fitted model.
            del self._X, self._y
        return self

    def _grow(self, indices: list[int], depth: int) -> TreeNode:
        labels = [self._y[i] for i in indices]
        counts = Counter(labels)
        majority = max(self.classes_, key=lambda c: counts[c])
        node = TreeNode(
            prediction=majority,
            samples=len(indices),
            counts=dict(counts),
            impurity=entropy(labels),
            depth=depth,
        )
        if (
            len(counts) == 1
            or len(indices) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return node

        best_gain = 0.0
        best: tuple[int, Any, bool, list[int], list[int]] | None = None
        for j in range(self.n_features_in_ or 0):
            values = {self._X[i][j] for i in indices}
            if len(values) < 2:
                continue
            if self.feature_is_numeric_[j]:
                ordered = sorted(values)
                candidates = [(a + (b - a) / 2, False) for a, b in zip(ordered, ordered[1:])]
            else:
                candidates = [(value, True) for value in sorted(values, key=repr)]
            for split_value, is_category in candidates:
                if is_category:
                    left = [i for i in indices if self._X[i][j] == split_value]
                    right = [i for i in indices if self._X[i][j] != split_value]
                else:
                    left = [i for i in indices if self._X[i][j] <= split_value]
                    right = [i for i in indices if self._X[i][j] > split_value]
                if min(len(left), len(right)) < self.min_samples_leaf:
                    continue
                gain = information_gain(labels, [self._y[i] for i in left], [self._y[i] for i in right])
                # Strict comparison gives deterministic first-found tie breaking.
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best = (j, split_value, is_category, left, right)

        if best is None or best_gain <= self.min_information_gain + 1e-12:
            return node
        j, split_value, categorical, left, right = best
        node.feature_index = j
        node.categorical = categorical
        node.category = split_value if categorical else None
        node.threshold = None if categorical else split_value
        node.gain = best_gain
        node.left = self._grow(left, depth + 1)
        node.right = self._grow(right, depth + 1)
        return node

    def _check_fitted(self) -> TreeNode:
        if self.root_ is None:
            raise RuntimeError("Call fit() before predicting or exporting a tree")
        return self.root_

    def predict_one(self, row: Sequence[Any]) -> Any:
        node = self._check_fitted()
        if len(row) != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {len(row)}")
        while not node.is_leaf:
            value = row[node.feature_index]  # type: ignore[index]
            if value is None:
                raise ValueError("Missing feature values are not supported")
            if node.categorical:
                go_left = value == node.category
            else:
                if not isinstance(value, Real) or not isfinite(value):
                    raise ValueError("A numerical feature must be a finite number")
                go_left = value <= node.threshold  # type: ignore[operator]
            node = node.left if go_left else node.right  # type: ignore[assignment]
        return node.prediction

    def predict(self, X: Sequence[Sequence[Any]]) -> list[Any]:
        """Return one predicted class label per input row."""
        return [self.predict_one(row) for row in X]

    def score(self, X: Sequence[Sequence[Any]], y: Sequence[Any]) -> float:
        """Return classification accuracy."""
        labels = list(y)
        predictions = self.predict(X)
        if not labels or len(labels) != len(predictions):
            raise ValueError("X and y must contain the same positive number of rows")
        return sum(a == b for a, b in zip(predictions, labels)) / len(labels)

    def get_depth(self) -> int:
        """Actual maximum depth of the fitted tree."""
        node = self._check_fitted()

        def depth(current: TreeNode) -> int:
            if current.is_leaf:
                return current.depth
            return max(depth(current.left), depth(current.right))  # type: ignore[arg-type]

        return depth(node)

    def get_n_leaves(self) -> int:
        """Actual number of leaf nodes."""
        node = self._check_fitted()

        def leaves(current: TreeNode) -> int:
            if current.is_leaf:
                return 1
            return leaves(current.left) + leaves(current.right)  # type: ignore[arg-type]

        return leaves(node)

    def _condition(self, node: TreeNode) -> str:
        name = self.feature_names_[node.feature_index]  # type: ignore[index]
        if node.categorical:
            return f"{name} == {node.category!r}"
        return f"{name} <= {node.threshold:.6g}"  # type: ignore[union-attr]

    def export_text(self) -> str:
        """Return a readable text drawing of the fitted tree."""
        lines = []

        def visit(node: TreeNode, prefix: str) -> None:
            if node.is_leaf:
                lines.append(f"{prefix}PREDICT {node.prediction!r} (n={node.samples}, H={node.impurity:.3f})")
                return
            lines.append(f"{prefix}IF {self._condition(node)} (gain={node.gain:.3f}, n={node.samples})")
            visit(node.left, prefix + "  YES -> ")  # type: ignore[arg-type]
            visit(node.right, prefix + "  NO  -> ")  # type: ignore[arg-type]

        visit(self._check_fitted(), "")
        return "\n".join(lines)

    def export_dot(self) -> str:
        """Return Graphviz DOT source (renderable with `dot -Tpng`)."""
        import json

        lines = ['digraph DecisionTree {', '  graph [rankdir=TB];',
                 '  node [shape=box, style="rounded,filled", fontname="Arial", fillcolor="#eaf3ff"];',
                 '  edge [fontname="Arial"];']
        counter = 0

        def visit(node: TreeNode) -> int:
            nonlocal counter
            identifier = counter
            counter += 1
            if node.is_leaf:
                label = f"Predict: {node.prediction!s}\nsamples: {node.samples}\nentropy: {node.impurity:.3f}"
                lines.append(f'  n{identifier} [label={json.dumps(label)}, fillcolor="#d7f5df"];')
            else:
                label = f"{self._condition(node)}\nsamples: {node.samples}\nentropy: {node.impurity:.3f}\ngain: {node.gain:.3f}"
                lines.append(f'  n{identifier} [label={json.dumps(label)}];')
                left_id = visit(node.left)  # type: ignore[arg-type]
                right_id = visit(node.right)  # type: ignore[arg-type]
                lines.extend([f'  n{identifier} -> n{left_id} [label="Yes"];',
                              f'  n{identifier} -> n{right_id} [label="No"];'])
            return identifier

        visit(self._check_fitted())
        lines.append('}')
        return "\n".join(lines) + "\n"

    def export_svg(self) -> str:
        """Create a directly viewable SVG tree, without optional packages."""
        from html import escape

        root = self._check_fitted()
        x_positions: dict[int, float] = {}
        leaves = 0

        def assign_x(node: TreeNode) -> float:
            nonlocal leaves
            if node.is_leaf:
                result = 145.0 + leaves * 275.0
                leaves += 1
            else:
                left_x = assign_x(node.left)  # type: ignore[arg-type]
                right_x = assign_x(node.right)  # type: ignore[arg-type]
                result = (left_x + right_x) / 2
            x_positions[id(node)] = result
            return result

        assign_x(root)
        width = max(420, 290 + (leaves - 1) * 275)
        height = 170 + self.get_depth() * 145
        elements = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#f8fafc"/>',
            '<style>text{font-family:Arial,sans-serif;fill:#172b4d} .main{font-size:14px;font-weight:600} .small{font-size:12px;fill:#52647c} .edge{stroke:#8493a6;stroke-width:2}</style>',
        ]

        def draw_edges(node: TreeNode) -> None:
            if node.is_leaf:
                return
            x1, y1 = x_positions[id(node)], 55 + node.depth * 145
            for branch, child in [("Yes", node.left), ("No", node.right)]:
                x2, y2 = x_positions[id(child)], 55 + child.depth * 145  # type: ignore[union-attr]
                elements.append(f'<line class="edge" x1="{x1}" y1="{y1+39}" x2="{x2}" y2="{y2-39}"/>')
                label_x = x1 + (x2 - x1) * 0.54
                label_y = y1 + (y2 - y1) * 0.54
                elements.append(f'<text class="small" x="{label_x}" y="{label_y-6}" text-anchor="middle">{branch}</text>')
                draw_edges(child)  # type: ignore[arg-type]

        def draw_nodes(node: TreeNode) -> None:
            x, y = x_positions[id(node)], 55 + node.depth * 145
            fill = "#dcfce7" if node.is_leaf else "#dbeafe"
            stroke = "#86c5a0" if node.is_leaf else "#93b4ed"
            elements.append(f'<rect x="{x-118}" y="{y-39}" width="236" height="78" rx="11" fill="{fill}" stroke="{stroke}"/>')
            if node.is_leaf:
                main = f"Predict: {node.prediction}"
                info = f"samples: {node.samples} | entropy: {node.impurity:.3f}"
            else:
                main = self._condition(node)
                info = f"gain: {node.gain:.3f} | samples: {node.samples}"
            elements.append(f'<text class="main" text-anchor="middle" x="{x}" y="{y-5}">{escape(main)}</text>')
            elements.append(f'<text class="small" text-anchor="middle" x="{x}" y="{y+19}">{escape(info)}</text>')
            if not node.is_leaf:
                draw_nodes(node.left)  # type: ignore[arg-type]
                draw_nodes(node.right)  # type: ignore[arg-type]

        draw_edges(root)
        draw_nodes(root)
        elements.append("</svg>")
        return "\n".join(elements) + "\n"
