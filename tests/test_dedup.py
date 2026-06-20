#!/usr/bin/env python3
"""
Tests for the stroke deduplication logic in converter/transform.py.

Run with:
    python -m unittest tests.test_dedup
"""
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from converter.transform import (  # noqa: E402
    deduplicate_strokes,
    _stroke_signature,
    _strokes_similar,
    DEDUP_POS_THRESHOLD,
    DEDUP_COLOR_THRESHOLD,
    DEDUP_RADIUS_THRESHOLD,
)


def make_action(x0=0.1, y0=0.2, x1=0.3, y1=0.4,
                r=0.5, g=0.6, b=0.7, a=0.8, radius=5.0):
    return {
        "x_start": x0, "y_start": y0, "x_end": x1, "y_end": y1,
        "color_r": r, "color_g": g, "color_b": b, "color_a": a,
        "brush_radius": radius,
    }


class TestStrokeSignature(unittest.TestCase):
    """Tests for _stroke_signature()."""

    def test_identical_strokes_same_signature(self):
        """Identical strokes should produce the same signature."""
        a = make_action()
        b = make_action()
        self.assertEqual(_stroke_signature(a), _stroke_signature(b))

    def test_different_position_different_signature(self):
        """Strokes at very different positions should have different signatures."""
        a = make_action(x0=0.1, y0=0.2)
        b = make_action(x0=0.9, y0=0.9)
        self.assertNotEqual(_stroke_signature(a), _stroke_signature(b))

    def test_different_color_different_signature(self):
        """Strokes with very different colours should have different signatures."""
        a = make_action(r=0.1, g=0.1, b=0.1)
        b = make_action(r=0.9, g=0.9, b=0.9)
        self.assertNotEqual(_stroke_signature(a), _stroke_signature(b))


class TestStrokesSimilar(unittest.TestCase):
    """Tests for _strokes_similar()."""

    def test_identical_strokes_similar(self):
        """Identical strokes should be similar."""
        a = make_action()
        b = make_action()
        self.assertTrue(_strokes_similar(a, b))

    def test_distant_strokes_not_similar(self):
        """Strokes far apart should not be similar."""
        a = make_action(x0=0.1, y0=0.1)
        b = make_action(x0=0.9, y0=0.9)
        self.assertFalse(_strokes_similar(a, b))

    def test_different_color_not_similar(self):
        """Strokes with very different colours should not be similar."""
        a = make_action(r=0.1, g=0.1, b=0.1)
        b = make_action(r=0.9, g=0.9, b=0.9)
        self.assertFalse(_strokes_similar(a, b))

    def test_different_radius_not_similar(self):
        """Strokes with very different radii should not be similar."""
        a = make_action(radius=2.0)
        b = make_action(radius=20.0)
        self.assertFalse(_strokes_similar(a, b))


class TestDeduplicateStrokes(unittest.TestCase):
    """Tests for deduplicate_strokes()."""

    def test_empty_list(self):
        """Empty list should return empty list."""
        self.assertEqual(deduplicate_strokes([]), [])

    def test_single_stroke(self):
        """Single stroke should be returned unchanged."""
        a = make_action()
        result = deduplicate_strokes([a])
        self.assertEqual(len(result), 1)

    def test_identical_strokes_deduped(self):
        """Identical strokes should be deduplicated."""
        a = make_action()
        actions = [a, a, a, a, a]
        result = deduplicate_strokes(actions)
        self.assertLess(len(result), 5)
        self.assertGreaterEqual(len(result), 1)

    def test_chain_merge(self):
        """Strokes where B starts where A ended should be merged."""
        # Stroke A: (0.1, 0.1) -> (0.2, 0.2)
        a = make_action(x0=0.1, y0=0.1, x1=0.2, y1=0.2, r=0.5, g=0.5, b=0.5, radius=5.0)
        # Stroke B: (0.2, 0.2) -> (0.3, 0.3) — same colour + radius
        b = make_action(x0=0.2, y0=0.2, x1=0.3, y1=0.3, r=0.5, g=0.5, b=0.5, radius=5.0)
        result = deduplicate_strokes([a, b])
        # Should merge into 1 stroke
        self.assertEqual(len(result), 1)
        # Merged stroke should end at (0.3, 0.3)
        self.assertAlmostEqual(result[0]["x_end"], 0.3, places=6)
        self.assertAlmostEqual(result[0]["y_end"], 0.3, places=6)

    def test_no_merge_different_color(self):
        """Strokes with different colours should not be merged."""
        a = make_action(x0=0.1, y0=0.1, x1=0.2, y1=0.2, r=0.1, g=0.1, b=0.1)
        b = make_action(x0=0.2, y0=0.2, x1=0.3, y1=0.3, r=0.9, g=0.9, b=0.9)
        result = deduplicate_strokes([a, b])
        self.assertEqual(len(result), 2)

    def test_no_merge_different_radius(self):
        """Strokes with very different radii should not be merged."""
        a = make_action(x0=0.1, y0=0.1, x1=0.2, y1=0.2, radius=2.0)
        b = make_action(x0=0.2, y0=0.2, x1=0.3, y1=0.3, radius=20.0)
        result = deduplicate_strokes([a, b])
        self.assertEqual(len(result), 2)

    def test_input_not_mutated(self):
        """The input list should not be mutated."""
        a = make_action()
        b = make_action()
        actions = [a, b]
        original_len = len(actions)
        deduplicate_strokes(actions)
        self.assertEqual(len(actions), original_len)

    def test_many_unique_strokes_preserved(self):
        """Many unique strokes should all be preserved."""
        actions = []
        for i in range(20):
            actions.append(make_action(
                x0=0.01 * i, y0=0.01 * i, x1=0.01 * i + 0.05, y1=0.01 * i + 0.05,
                r=0.05 * i % 1.0, radius=2.0 + 0.5 * i
            ))
        result = deduplicate_strokes(actions)
        self.assertEqual(len(result), 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
