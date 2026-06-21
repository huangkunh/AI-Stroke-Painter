#!/usr/bin/env python3
"""
Tests for the API warm-up mechanism and CDN/deployment configuration.

Run with:
    python -m unittest tests.test_warmup
"""
import json
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
API_DIR = os.path.join(PROJECT_ROOT, "api")
if API_DIR not in sys.path:
    sys.path.insert(0, API_DIR)


class TestWarmupMechanism(unittest.TestCase):
    """Tests for the API warm-up mechanism in api/infer.py."""

    def test_warmup_completes_without_error(self):
        """The warmup function should complete without raising."""
        import infer
        # Warmup runs at module import; just verify it completed
        self.assertTrue(infer._warmup_done,
                        "Warmup should be done after module import")
        self.assertIsNone(infer._warmup_error,
                          f"Warmup error: {infer._warmup_error}")

    def test_warmup_is_idempotent(self):
        """Calling warmup twice should not re-run."""
        import infer
        # Already done from module import
        infer._warmup()
        self.assertTrue(infer._warmup_done)

    def test_warmup_imports_numpy(self):
        """After warmup, numpy should be importable (cached)."""
        import infer
        self.assertTrue(infer._warmup_done)
        import numpy as np
        self.assertIsNotNone(np)


class TestVercelConfig(unittest.TestCase):
    """Tests for the vercel.json deployment configuration."""

    @classmethod
    def setUpClass(cls):
        cls.config_path = os.path.join(PROJECT_ROOT, "vercel.json")
        with open(cls.config_path, "r", encoding="utf-8") as f:
            cls.config = json.load(f)

    def test_config_has_regions(self):
        """vercel.json should configure multi-region deployment."""
        self.assertIn("regions", self.config,
                      "Missing 'regions' in vercel.json")
        regions = self.config["regions"]
        self.assertGreaterEqual(len(regions), 3,
                                f"Should have >=3 regions, got {len(regions)}")
        # Check for key regions
        region_str = " ".join(regions)
        self.assertIn("sfo1", region_str, "Missing US West region")
        self.assertIn("iad1", region_str, "Missing US East region")

    def test_config_has_cdn_headers(self):
        """vercel.json should configure CDN cache headers."""
        self.assertIn("headers", self.config)
        headers = self.config["headers"]
        # Find JS cache header
        js_header = None
        for h in headers:
            if ".js" in h.get("source", ""):
                js_header = h
                break
        self.assertIsNotNone(js_header, "Missing JS cache header config")
        # Check for immutable cache directive
        cache_values = [hk["value"] for hk in js_header["headers"]
                        if hk["key"] == "Cache-Control"]
        self.assertTrue(any("immutable" in v for v in cache_values),
                        "JS cache should be immutable")

    def test_config_has_function_config(self):
        """vercel.json should configure function memory and timeout."""
        self.assertIn("functions", self.config)
        funcs = self.config["functions"]
        self.assertIn("api/infer.py", funcs)
        infer_cfg = funcs["api/infer.py"]
        self.assertIn("memory", infer_cfg)
        self.assertIn("maxDuration", infer_cfg)
        self.assertGreaterEqual(infer_cfg["memory"], 512,
                                "infer function should have >=512MB memory")
        self.assertGreaterEqual(infer_cfg["maxDuration"], 30,
                               "infer function should have >=30s timeout")

    def test_config_has_api_routes(self):
        """vercel.json should have routes for all API endpoints."""
        self.assertIn("routes", self.config)
        routes = self.config["routes"]
        api_routes = [r["src"] for r in routes if "/api/" in r.get("src", "")]
        self.assertIn("/api/infer", api_routes, "Missing /api/infer route")
        self.assertIn("/api/transform", api_routes, "Missing /api/transform route")
        self.assertIn("/api/pipeline", api_routes, "Missing /api/pipeline route")

    def test_config_has_cors_headers(self):
        """vercel.json should configure CORS for API endpoints."""
        headers = self.config["headers"]
        api_header = None
        for h in headers:
            if "/api/" in h.get("source", ""):
                api_header = h
                break
        self.assertIsNotNone(api_header, "Missing API CORS header config")
        cors_values = [hk["value"] for hk in api_header["headers"]
                       if hk["key"] == "Access-Control-Allow-Origin"]
        self.assertTrue(len(cors_values) > 0, "Missing CORS header")


if __name__ == "__main__":
    unittest.main(verbosity=2)
