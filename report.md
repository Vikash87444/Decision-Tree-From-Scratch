# Decision Tree Algorithm — Assignment Report

**Student:** Vikash Sakthivel  
**Subject:** Machine Learning / Decision Tree Algorithm  
**Submission:** 26 October 2026  
**GitHub repository:** **[ADD YOUR GITHUB URL AFTER PUSHING THIS PROJECT]**

## 1. Objective

I implemented my own decision tree classifier **from scratch** using Python, with **entropy** to measure uncertainty and **information gain** to choose the best split. I also added a hyperparameter to control how large the tree can grow. The implementation does not call a prebuilt decision-tree classifier.

## 2. Methodology

Entropy measures how mixed the class labels are:

`H(S) = - sum(p(class) * log2(p(class)))`

Information gain compares entropy before and after splitting:

`IG = H(parent) - (n_left / n) * H(left) - (n_right / n) * H(right)`

For each training node, I check candidate splits for every feature. Numeric features use `<= threshold` and categorical features use `== category`. I select the split yielding the **highest positive information gain**, then repeat the procedure for the child nodes. Each terminal node predicts the majority class of its training observations.

## 3. Hyperparameters

The main tree-size hyperparameter is `max_depth` (default `4`). A setting of `0` creates one majority-class leaf; `2` limits the tree to two splitting levels; `None` removes the explicit depth limit.

I also implemented `min_samples_split`, `min_samples_leaf`, and `min_information_gain`. These control when splitting is allowed and help limit excessive growth. Tests verify that the constraints are honored.

## 4. Demonstration and evaluation

The `demo.py` program contains two examples:

1. **Play Tennis:** a small 14-row dataset with four categorical attributes. It demonstrates how equality splits make a readable classification tree.
2. **Synthetic numerical dataset:** 240 reproducibly generated points with two features. A randomized but deterministic split keeps 168 points for training and 72 points for **held-out testing**. I report training accuracy and test accuracy separately to avoid confusing memorization with generalization.

Run `python demo.py` to see the actual accuracy, depth, number of leaves and split-by-split tree. Run `python demo.py --max-depth 2` to see how limiting tree size affects the model.

**Measured default-run results (Python 3.12, `max_depth=4`):**

| Dataset | Training accuracy | Held-out test accuracy | Tree depth | Number of leaves |
|---|---:|---:|---:|---:|
| Play Tennis (categorical) | 100.00% | Not applicable | 4 | 7 |
| Synthetic numeric | 100.00% | 98.61% | 2 | 3 |

**Effect of the size hyperparameter:** with `max_depth=2`, Play Tennis training accuracy falls to **85.71%** and its tree has only **3 leaves** instead of 7. This is an example of underfitting when a model is too constrained. For the synthetic numeric rule, a depth of 2 is already sufficient, so its held-out score remains **98.61%**.

On a manually verified example, four parent labels `[A, A, B, B]` have entropy **1 bit**. Splitting into `[A, A]` and `[B, B]` gives **0 entropy** in either child; the information gain is **1 bit**. A unit test verifies this result.

## 5. Visualization bonus

The classifier supports `export_text()`, `export_svg()` and `export_dot()`:

- `export_text()` shows every test and leaf prediction directly in the terminal.
- `export_svg()` saves a complete visual tree as a standard `.svg` image that opens directly in a browser, without third-party libraries.
- `export_dot()` generates Graphviz `.dot` files under `outputs/`. With Graphviz installed, the user can render those as PNG diagrams using `dot -Tpng`.

## 6. Testing

I wrote unit tests covering entropy, information gain, numerical and categorical splits, depth and sample constraints, model predictions, accuracy, malformed input, and DOT export.

Run:

`python -m unittest discover -s tests -v`

**Test status:** All **18** automated tests passed on Python 3.12, including SVG export.

## 7. Bonus: packaging

The project contains a `pyproject.toml` defining an installable package, and can be installed locally with `python -m pip install .`. It has **not** been published to PyPI. Actual deployment would require account credentials and uploading a release.

## 8. Limitations and possible improvements

This is a learning-oriented implementation: it uses binary classification splits, requires complete input data, and checks candidate splits greedily. Potential next steps include missing-value handling, reduced-error pruning, cross-validation, performance optimization, and safe model serialization.

## 9. Files included

`decision_tree.py`, `demo.py`, `tests/test_decision_tree.py`, `README.md`, `pyproject.toml`, and this `report.md`.
