import unittest

import pkps_theme


class ThemeTests(unittest.TestCase):
    def test_asset_is_clean_svg(self):
        s = pkps_theme.ASSET.read_text(encoding="utf-8")
        self.assertTrue(s.lstrip().startswith("<svg"))
        import base64, re
        uri = re.search(r"base64,([A-Za-z0-9+/=]+)", pkps_theme.build_css()).group(1)
        self.assertNotIn(b"<metadata", base64.b64decode(uri))

    def test_css_embeds_background(self):
        css = pkps_theme.build_css()
        self.assertIn("data:image/svg+xml;base64,", css)
        self.assertEqual(css.count("<style>"), 1)
        self.assertEqual(css.count("{"), css.count("}"))

    def test_css_without_asset(self):
        old = pkps_theme.ASSET
        try:
            pkps_theme.ASSET = old.with_name("nope.svg")
            self.assertNotIn("data:image", pkps_theme.build_css())
        finally:
            pkps_theme.ASSET = old


if __name__ == "__main__":
    unittest.main()
