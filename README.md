# Decision Tree Classifier from Scratch

**Assignment:** Implement a decision tree using Shannon entropy and information gain, with a tree-size hyperparameter. Optional visualization and PyPI packaging.

This project implements a binary-splitting **classification tree in pure Python**, without importing a decision-tree implementation or using `scikit-learn` for training. It supports numerical and categorical features and generates readable tree visualizations.

## Run

Requires Python 3.10 or later. No third-party Python packages are required.

```bash
python demo.py
python demo.py --max-depth 2 --output-dir outputs_depth2
python demo.py --max-depth 5 --min-samples-leaf 2
python -m unittest discover -s tests -v
```

The demo creates `outputs/play_tennis_tree.svg` and `outputs/numeric_tree.svg`, which can be opened directly in a web browser, **without any extra packages**. It also creates Graphviz `.dot` files. To turn a DOT file into a PNG, install Graphviz separately and run:

```bash
dot -Tpng outputs/play_tennis_tree.dot -o outputs/play_tennis_tree.png
dot -Tpng outputs/numeric_tree.dot -o outputs/numeric_tree.png
```

`export_text()` also prints the full tree in your terminal.

## How the algorithm works

For a label set \(S\), entropy measures class uncertainty:

\[
H(S) = -\sum_{c}p(c)\log_2 p(c)
\]

For a candidate split that produces left and right groups:

\[
IG = H(S) - \frac{|S_L|}{|S|}H(S_L) - \frac{|S_R|}{|S|}H(S_R)
\]

At each node, the tree:

1. Tests thresholds halfway between distinct values of each **numeric** feature.
2. Tests equality against each category for each **categorical** feature.
3. Chooses the binary split with the largest information gain, while respecting `min_samples_leaf`.
4. Stops at a pure leaf, `max_depth`, `min_samples_split`, or when no split has sufficient positive information gain.
5. Predicts the majority class at each terminal leaf; ties are resolved consistently.

## Tree-size hyperparameters

| Hyperparameter | Default | Meaning |
|---|---:|---|
| `max_depth` | `4` | Maximum levels of splits, root depth = 0; `None` removes the limit |
| `min_samples_split` | `2` | Minimum examples needed to attempt a node split |
| `min_samples_leaf` | `1` | Minimum examples on either side of a split |
| `min_information_gain` | `0.0` | Minimum improvement needed to allow a split |

Smaller values of `max_depth` produce simpler trees that can underfit; larger values can fit training data more closely but may overfit.

## Python API

```python
from decision_tree import DecisionTreeClassifier

X = [[20, "Sunny"], [21, "Sunny"], [28, "Rain"], [30, "Rain"]]
y = ["No", "No", "Yes", "Yes"]

model = DecisionTreeClassifier(max_depth=2, min_samples_leaf=1)
model.fit(X, y, feature_names=["Temperature", "Weather"])
print(model.predict([[29, "Rain"], [19, "Sunny"]]))
print(model.score(X, y))
print(model.export_text())
print(model.get_depth(), model.get_n_leaves())
```

## Files

- `decision_tree.py` — entropy, information gain, node representation, training, prediction, scoring and visualizations.
- `demo.py` — categorical Play Tennis example, synthetic numerical train/test example, and generated SVG/DOT files.
- `tests/test_decision_tree.py` — automated unit tests for formulas, splits, limits, validation, predictions and DOT export.
- `report.md` — assignment write-up (add your GitHub repository link after pushing).
- `pyproject.toml` — packaging metadata so the project can be built or locally installed; **not published to PyPI**.

## Optional packaging bonus

For a local installation, from this directory:

```bash
python -m pip install .
```

For building an archive (requires Python's separate `build` package):

```bash
python -m pip install build
python -m build
```

Publication to PyPI is an additional step requiring a PyPI account and explicit decision to publish. This submission contains **packaging preparation only**, not a claim of being deployed on PyPI.

## Limitations

- Classification only, not regression.
- Binary splitting; categorical values are tested one category against the rest.
- Missing data and non-finite numeric values are rejected explicitly.
- Greedy splitting searches candidate values independently; it is educational and not optimized for very large datasets.
- Demo data is illustrative; no general real-world performance claim is made.
