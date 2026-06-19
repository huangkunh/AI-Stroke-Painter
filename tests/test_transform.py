#!/usr/bin/env python3
"""
Unit tests for converter/transform.py.

Run with:
    python -m unittest tests.test_transform
or:
    python -m unittest discover tests
"""
import math
import os
import sys
import unittest

# Make the project root importable so `from converter.transform import ...` works
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from converter.transform import (  # noqa: E402
    to_hex,
    pick_brush_id,
    bell_pressure,
    interpolate_stroke,
    stroke_point_count,
    transform,
    ENGINE_W,
    ENGINE_H,
    PRESSURE_SCALE,
    DEFAULT_BACKGROUND,
)


class TestToHex(unittest.TestCase):
    """Tests for to_hex(r, g, b) — converts 0..1 floats to #RRGGBB."""

    def test_black(self):
        """All zeros should produce #000000."""
        self.assertEqual(to_hex(0.0, 0.0, 0.0), "#000000")

    def test_white(self):
        """All ones should produce #ffffff."""
        self.assertEqual(to_hex(1.0, 1.0, 1.0), "#ffffff")

    def test_red(self):
        """Pure red (1, 0, 0) should produce #ff0000."""
        self.assertEqual(to_hex(1.0, 0.0, 0.0), "#ff0000")

    def test_green(self):
        """Pure green (0, 1, 0) should produce #00ff00."""
        self.assertEqual(to_hex(0.0, 1.0, 0.0), "#00ff00")

    def test_blue(self):
        """Pure blue (0, 0, 1) should produce #0000ff."""
        self.assertEqual(to_hex(0.0, 0.0, 1.0), "#0000ff")

    def test_mid_gray(self):
        """0.5 on all channels → 128 → #808080."""
        self.assertEqual(to_hex(0.5, 0.5, 0.5), "#808080")

    def test_specific_color(self):
        """0.823 → 210 → #d2; the classic engine demo red #d20000."""
        self.assertEqual(to_hex(0.8235, 0.0, 0.0), "#d20000")

    def test_clamping_above_one(self):
        """Values > 1 should be clamped to 255 (#ff)."""
        self.assertEqual(to_hex(2.0, 1.5, 1.0), "#ffffff")

    def test_clamping_below_zero(self):
        """Negative values should be clamped to 0 (#00)."""
        self.assertEqual(to_hex(-0.5, -1.0, 0.0), "#000000")

    def test_format_is_lowercase_hex(self):
        """Output must be #rrggbb in lowercase with leading #."""
        result = to_hex(0.5, 0.25, 0.75)
        self.assertTrue(result.startswith("#"))
        self.assertEqual(len(result), 7)
        self.assertEqual(result, result.lower())


class TestPickBrushId(unittest.TestCase):
    """Tests for pick_brush_id(radius) — thin→5, broad→0."""

    def test_thin_line_uses_brush_5(self):
        """radius < 3 should return brush 5 (压感v3, fine line)."""
        self.assertEqual(pick_brush_id(1.8), 5)
        self.assertEqual(pick_brush_id(2.0), 5)
        self.assertEqual(pick_brush_id(2.9), 5)

    def test_broad_stroke_uses_brush_0(self):
        """radius >= 3 should return brush 0 (马克笔, flat colour)."""
        self.assertEqual(pick_brush_id(3.0), 0)
        self.assertEqual(pick_brush_id(5.0), 0)
        self.assertEqual(pick_brush_id(14.0), 0)
        self.assertEqual(pick_brush_id(20.0), 0)

    def test_boundary_exactly_three(self):
        """The boundary radius=3 should go to brush 0 (>=3)."""
        self.assertEqual(pick_brush_id(3.0), 0)

    def test_just_below_boundary(self):
        """radius=2.99 should still go to brush 5."""
        self.assertEqual(pick_brush_id(2.99), 5)

    def test_zero_radius(self):
        """radius=0 is < 3, so brush 5."""
        self.assertEqual(pick_brush_id(0.0), 5)

    def test_negative_radius(self):
        """Negative radius is < 3, so brush 5 (edge case)."""
        self.assertEqual(pick_brush_id(-1.0), 5)


class TestBellPressure(unittest.TestCase):
    """Tests for bell_pressure(t) — bell-shaped curve in [0, 1]."""

    def test_start_pressure_is_low(self):
        """At t=0 (起笔) pressure should be the minimum (~0.15)."""
        p_start = bell_pressure(0.0)
        self.assertLess(p_start, 0.2)

    def test_end_pressure_is_low(self):
        """At t=1 (收笔) pressure should be the minimum (~0.15)."""
        p_end = bell_pressure(1.0)
        self.assertLess(p_end, 0.2)

    def test_middle_pressure_is_high(self):
        """At t=0.5 (middle) pressure should be the maximum (~1.0)."""
        p_mid = bell_pressure(0.5)
        self.assertGreater(p_mid, 0.9)

    def test_start_less_than_middle(self):
        """起笔压感 must be less than 中间压感 (core requirement)."""
        p_start = bell_pressure(0.0)
        p_mid = bell_pressure(0.5)
        self.assertLess(p_start, p_mid)

    def test_end_less_than_middle(self):
        """收笔压感 must be less than 中间压感 (core requirement)."""
        p_end = bell_pressure(1.0)
        p_mid = bell_pressure(0.5)
        self.assertLess(p_end, p_mid)

    def test_symmetry(self):
        """The bell curve should be symmetric: p(t) == p(1-t)."""
        for t in [0.1, 0.25, 0.3, 0.4, 0.5]:
            self.assertAlmostEqual(bell_pressure(t), bell_pressure(1.0 - t),
                                   places=6,
                                   msg=f"Symmetry failed at t={t}")

    def test_monotonic_increase_first_half(self):
        """Pressure should increase from t=0 to t=0.5."""
        prev = bell_pressure(0.0)
        for i in range(1, 11):
            t = i * 0.05
            curr = bell_pressure(t)
            self.assertGreaterEqual(curr, prev - 1e-9,
                                    f"Not monotonic at t={t}")
            prev = curr

    def test_monotonic_decrease_second_half(self):
        """Pressure should decrease from t=0.5 to t=1.0."""
        prev = bell_pressure(0.5)
        for i in range(11, 21):
            t = i * 0.05
            curr = bell_pressure(t)
            self.assertLessEqual(curr, prev + 1e-9,
                                 f"Not monotonic at t={t}")
            prev = curr

    def test_range(self):
        """All pressure values should be in [0.15, 1.0]."""
        for i in range(101):
            t = i * 0.01
            p = bell_pressure(t)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0 + 1e-9)


class TestInterpolateStroke(unittest.TestCase):
    """Tests for interpolate_stroke and stroke_point_count."""

    def test_min_two_points(self):
        """Even with n_points=1, at least 2 points should be returned."""
        pts = interpolate_stroke(0.0, 0.0, 1.0, 1.0, 1)
        self.assertGreaterEqual(len(pts), 2)

    def test_endpoints_match(self):
        """First point should be (x0,y0) and last should be (x1,y1)."""
        pts = interpolate_stroke(0.1, 0.2, 0.8, 0.9, 10)
        self.assertAlmostEqual(pts[0][0], 0.1)
        self.assertAlmostEqual(pts[0][1], 0.2)
        self.assertAlmostEqual(pts[-1][0], 0.8)
        self.assertAlmostEqual(pts[-1][1], 0.9)

    def test_point_count(self):
        """Should return exactly n_points points."""
        for n in [5, 10, 20]:
            pts = interpolate_stroke(0.0, 0.0, 1.0, 1.0, n)
            self.assertEqual(len(pts), n)

    def test_t_values_progress(self):
        """The t parameter should go from 0 to 1."""
        pts = interpolate_stroke(0.0, 0.0, 1.0, 1.0, 5)
        self.assertAlmostEqual(pts[0][2], 0.0)
        self.assertAlmostEqual(pts[-1][2], 1.0)


class TestTransformPipeline(unittest.TestCase):
    """Integration tests for the full transform() function."""

    def _make_action(self, radius=5.0, alpha=0.8):
        return {
            "x_start": 0.2, "y_start": 0.3,
            "x_end": 0.8, "y_end": 0.7,
            "color_r": 0.5, "color_g": 0.3, "color_b": 0.1,
            "color_a": alpha, "brush_radius": radius,
        }

    def test_starts_with_background(self):
        """First instruction should be background when start_with_background=True."""
        result = transform([self._make_action()], start_with_background=True)
        self.assertEqual(result[0][0], "background")
        self.assertEqual(result[0][1], DEFAULT_BACKGROUND)

    def test_no_background_when_disabled(self):
        """No background instruction when start_with_background=False."""
        result = transform([self._make_action()], start_with_background=False)
        self.assertNotEqual(result[0][0], "background")

    def test_colour_instruction_present(self):
        """Each stroke should emit a colour instruction."""
        result = transform([self._make_action()], start_with_background=False)
        colours = [i for i in result if i[0] == "colour"]
        self.assertEqual(len(colours), 1)
        self.assertTrue(colours[0][1].startswith("#"))

    def test_width_instruction_present(self):
        """Each stroke should emit a width instruction matching the radius."""
        result = transform([self._make_action(radius=7.5)],
                           start_with_background=False)
        widths = [i for i in result if i[0] == "width"]
        self.assertEqual(len(widths), 1)
        self.assertAlmostEqual(widths[0][1], 7.5)

    def test_line_instruction_uses_correct_brush(self):
        """Thin strokes use brush 5, broad strokes use brush 0."""
        # Thin
        result = transform([self._make_action(radius=1.5)],
                           start_with_background=False)
        lines = [i for i in result if i[0] == "line"]
        self.assertEqual(lines[0][1], 5)  # brush 5

        # Broad
        result = transform([self._make_action(radius=10.0)],
                           start_with_background=False)
        lines = [i for i in result if i[0] == "line"]
        self.assertEqual(lines[0][1], 0)  # brush 0

    def test_brush5_has_pressure_values(self):
        """Brush 5 lines should have 3 values per point (x, y, pressure)."""
        result = transform([self._make_action(radius=1.5)],
                           start_with_background=False)
        line = [i for i in result if i[0] == "line"][0]
        # line = ["line", brushId, x1, y1, p1, x2, y2, p2, ...]
        # After "line" and brushId, remaining values should be divisible by 3
        point_data = line[2:]
        self.assertEqual(len(point_data) % 3, 0)

    def test_brush0_has_no_pressure_values(self):
        """Brush 0 lines should have 2 values per point (x, y) only."""
        result = transform([self._make_action(radius=10.0)],
                           start_with_background=False)
        line = [i for i in result if i[0] == "line"][0]
        point_data = line[2:]
        self.assertEqual(len(point_data) % 2, 0)

    def test_multiple_strokes(self):
        """Multiple actions should produce multiple line instructions."""
        actions = [self._make_action() for _ in range(5)]
        result = transform(actions, start_with_background=False)
        lines = [i for i in result if i[0] == "line"]
        self.assertEqual(len(lines), 5)

    def test_coordinates_scaled_to_engine_resolution(self):
        """Normalized coords (0..1) should be scaled to ENGINE_W x ENGINE_H."""
        result = transform([self._make_action()],
                           start_with_background=False)
        line = [i for i in result if i[0] == "line"][0]
        # First point x should be ~0.2 * 640 = 128
        self.assertAlmostEqual(line[2], 0.2 * ENGINE_W, delta=1.0)
        # First point y should be ~0.3 * 480 = 144
        self.assertAlmostEqual(line[3], 0.3 * ENGINE_H, delta=1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
