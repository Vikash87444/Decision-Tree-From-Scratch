"""Automated checks; run `python -m unittest discover -s tests -v`."""

import math
import unittest

from decision_tree import DecisionTreeClassifier, entropy, information_gain
from demo import play_tennis_dataset, synthetic_numeric_dataset


class EntropyTests(unittest.TestCase):
    def test_entropy_empty_and_pure(self):
        self.assertEqual(entropy([]), 0.0)
        self.assertEqual(entropy(["yes"] * 5), 0.0)

    def test_entropy_balanced(self):
        self.assertAlmostEqual(entropy([0, 0, 1, 1]), 1.0)

    def test_entropy_unequal(self):
        self.assertAlmostEqual(entropy([0, 0, 0, 1]), 0.8112781244591328)

    def test_information_gain_perfect_and_zero(self):
        self.assertAlmostEqual(information_gain([0, 0, 1, 1], [0, 0], [1, 1]), 1.0)
        self.assertAlmostEqual(information_gain([0, 0, 1, 1], [0, 1], [0, 1]), 0.0)

    def test_information_gain_invalid_partition(self):
        with self.assertRaises(ValueError):
            information_gain([0, 1], [0], [])


class DecisionTreeTests(unittest.TestCase):
    def test_numerical_perfect_separation(self):
        X = [[1], [2], [3], [7], [8], [9]]
        y = ["L", "L", "L", "R", "R", "R"]
        model = DecisionTreeClassifier(max_depth=1).fit(X, y, ["value"])
        self.assertEqual(model.predict([[0], [4], [6], [10]]), ["L", "L", "R", "R"])
        self.assertEqual(model.get_depth(), 1)
        self.assertEqual(model.get_n_leaves(), 2)

    def test_categorical_play_tennis(self):
        X, y, names = play_tennis_dataset()
        model = DecisionTreeClassifier(max_depth=6).fit(X, y, names)
        self.assertEqual(model.score(X, y), 1.0)
        self.assertIn("Outlook", model.export_text())

    def test_max_depth_zero_is_one_leaf(self):
        model = DecisionTreeClassifier(max_depth=0).fit([[0], [1], [2]], ["A", "A", "B"])
        self.assertEqual(model.get_depth(), 0)
        self.assertEqual(model.get_n_leaves(), 1)
        self.assertEqual(model.predict([[100], [-100]]), ["A", "A"])

    def test_max_depth_bound(self):
        X, y, _ = play_tennis_dataset()
        model = DecisionTreeClassifier(max_depth=2).fit(X, y)
        self.assertLessEqual(model.get_depth(), 2)

    def test_min_samples_leaf_respected(self):
        model = DecisionTreeClassifier(min_samples_leaf=3).fit([[1], [2], [3], [4]], [0, 0, 0, 1])
        self.assertEqual(model.get_n_leaves(), 1)

    def test_min_samples_split_respected(self):
        model = DecisionTreeClassifier(min_samples_split=5).fit([[1], [2], [3], [4]], [0, 0, 1, 1])
        self.assertEqual(model.get_n_leaves(), 1)

    def test_no_positive_gain(self):
        model = DecisionTreeClassifier().fit([["a"], ["a"], ["b"], ["b"]], [0, 1, 0, 1])
        self.assertEqual(model.get_n_leaves(), 1)

    def test_valid_dot_export(self):
        model = DecisionTreeClassifier().fit([[1], [2], [5], [6]], ["A", "A", "B", "B"])
        dot = model.export_dot()
        self.assertIn("digraph DecisionTree", dot)
        self.assertIn('label="Yes"', dot)
        self.assertIn("Predict:", dot)

    def test_svg_visualization(self):
        model = DecisionTreeClassifier().fit([[1], [2], [5], [6]], ["A", "A", "B", "B"])
        svg = model.export_svg()
        self.assertIn("<svg", svg)
        self.assertIn("Predict: A", svg)
        self.assertIn("Predict: B", svg)
        self.assertIn("</svg>", svg)

    def test_score_on_holdout(self):
        X_train, y_train, X_test, y_test = synthetic_numeric_dataset()
        model = DecisionTreeClassifier(max_depth=4).fit(X_train, y_train)
        self.assertGreaterEqual(model.score(X_test, y_test), 0.9)

    def test_invalid_parameters(self):
        for kwargs in [{"max_depth": -1}, {"min_samples_split": 1},
                       {"min_samples_leaf": 0}, {"min_information_gain": -0.1}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DecisionTreeClassifier(**kwargs)

    def test_bad_training_inputs(self):
        cases = [([], []), ([[1], [2]], [1]), ([[1], [2, 3]], [1, 1]),
                 ([[1], [math.nan]], [1, 2]), ([[1], [None]], [1, 2])]
        for X, y in cases:
            with self.subTest(X=X), self.assertRaises(ValueError):
                DecisionTreeClassifier().fit(X, y)

    def test_predict_before_fit_and_wrong_width(self):
        model = DecisionTreeClassifier()
        with self.assertRaises(RuntimeError):
            model.predict([[1]])
        model.fit([[0], [1]], [0, 1])
        with self.assertRaises(ValueError):
            model.predict([[0, 1]])


if __name__ == "__main__":
    unittest.main()
