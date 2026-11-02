"""
Тесты новых модулей: высота борта и пропорции (ядро), геометрия корпуса,
ход решения, страница 3D-просмотра.

Запуск:  python -m unittest -v
"""
import json
import re
import unittest

import numpy as np

import pkps_core as core
import pkps_hull as hull_mod
import pkps_steps as steps
import pkps_view3d as view3d
from pkps_core import BasinClass as BC, ShipType as ST


def sample(**kw) -> core.ShipInput:
    base = dict(basin=BC.O, ship_type=ST.CARGO, L=110.0, B=16.5, T=3.5, Tn=3.4,
                Cb=0.85, V0_kmh=18.0, I=5.0, D=None, H=5.0)
    base.update(kw)
    return core.ShipInput(**base)


def hull_body(h: hull_mod.Hull3D) -> np.ndarray:
    """Вершины корпуса без надстройки (она выступает над палубой)."""
    idx = np.unique(np.concatenate([h.groups[k].ravel() for k in ("under", "above", "deck")]))
    return h.verts[idx]


# --------------------------------------------------------------------------
class TestCoreDepth(unittest.TestCase):
    def test_draft_stern(self):
        self.assertAlmostEqual(sample(T=3.5, Tn=3.0).T_stern, 4.0)

    def test_draft_above_depth_is_error(self):
        with self.assertRaises(core.InputError):
            core.calculate(sample(T=5.5, Tn=5.0, H=5.0))
        with self.assertRaises(core.InputError):
            core.calculate(sample(T=4.0, Tn=5.5, H=5.0))

    def test_ratio_warning_L_over_H(self):
        r = core.calculate(sample(H=3.6))          # L/H = 30,6 > 27 («О», табл. 2.1.3-1)
        self.assertTrue(any("L/H" in w for w in r.warnings))

    def test_ratio_warning_B_over_H(self):
        r = core.calculate(sample(H=3.5, T=3.0, Tn=3.0))   # B/H = 4,7 < 5 для «О», но L/H=31 > 27
        self.assertTrue(any("L/H" in w for w in r.warnings))
        r2 = core.calculate(sample(basin=BC.M, L=60.0, B=16.5, H=3.8, T=3.0, Tn=3.0))  # B/H = 4,34 > 4,0
        self.assertTrue(any("B/H" in w for w in r2.warnings))

    def test_no_ratio_warning_for_normal_ship(self):
        r = core.calculate(sample())
        self.assertFalse(any("L/H" in w or "B/H" in w for w in r.warnings))

    def test_H_optional(self):
        r = core.calculate(sample(H=None))
        self.assertGreater(r.Mv, 0)

    def test_H_does_not_change_moments(self):
        a = core.calculate(sample(H=5.0))
        b = core.calculate(sample(H=7.0))
        self.assertAlmostEqual(a.Mdv, b.Mdv, places=6)

    def test_table_lookups_match_rules(self):
        self.assertAlmostEqual(core.table_k_delta(0.85), 0.787)
        self.assertAlmostEqual(core.table_k_t(BC.O, 1.0), 0.805)
        self.assertAlmostEqual(core.table_k_b(BC.R, 12.0), 0.408)
        self.assertAlmostEqual(core.table_k_t(BC.M, 0.75), 0.910, places=3)   # интерполяция 0,5…1,0

    def test_table_lookup_out_of_range(self):
        self.assertIsNone(core.table_k_delta(0.4))
        self.assertIsNone(core.table_k_t(BC.M, 6.0))
        self.assertIsNone(core.table_k_b(BC.M, 1.0))


# --------------------------------------------------------------------------
class TestHull(unittest.TestCase):
    def test_bbox_matches_inputs(self):
        h = hull_mod.build_hull(110, 16.5, 5.0, 3.5, 3.5, 0.85)      # без дифферента
        v = hull_body(h)
        self.assertAlmostEqual(v[:, 0].min(), 0.0, places=6)
        self.assertAlmostEqual(v[:, 0].max(), 110.0, places=6)
        self.assertAlmostEqual(v[:, 1].max(), 8.25, places=6)
        self.assertAlmostEqual(v[:, 1].min(), -8.25, places=6)
        self.assertAlmostEqual(v[:, 2].max(), 5.0, places=6)
        self.assertAlmostEqual(v[:, 2].min(), 0.0, places=6)

    def test_dimensions_scale_with_inputs(self):
        a = hull_body(hull_mod.build_hull(60, 12, 4.0, 2.5, 2.5, 0.8))
        b = hull_body(hull_mod.build_hull(120, 12, 4.0, 2.5, 2.5, 0.8))
        self.assertAlmostEqual(np.ptp(b[:, 0]) / np.ptp(a[:, 0]), 2.0, places=6)
        c = hull_body(hull_mod.build_hull(60, 24, 4.0, 2.5, 2.5, 0.8))
        self.assertAlmostEqual(np.ptp(c[:, 1]) / np.ptp(a[:, 1]), 2.0, places=6)
        d = hull_body(hull_mod.build_hull(60, 12, 8.0, 2.5, 2.5, 0.8))
        self.assertAlmostEqual(np.ptp(d[:, 2]) / np.ptp(a[:, 2]), 2.0, places=6)

    def test_waterline_lies_on_draft(self):
        for tn in (3.5, 3.0, 4.2):
            h = hull_mod.build_hull(110, 16.5, 5.0, 3.5, tn, 0.85)
            self.assertAlmostEqual(h.waterline[:, 2].min(), 3.5, places=6)
            self.assertAlmostEqual(h.waterline[:, 2].max(), 3.5, places=6)

    def test_trim_direction(self):
        level = hull_mod.build_hull(110, 16.5, 5.0, 3.5, 3.5, 0.85)
        bow_down = hull_mod.build_hull(110, 16.5, 5.0, 3.5, 4.5, 0.85)
        self.assertEqual(level.trim_deg, 0.0)
        self.assertGreater(bow_down.trim_deg, 0.0)
        self.assertAlmostEqual(bow_down.Tk, 2.5)

        def bow_keel_z(h):
            v = hull_body(h)
            return v[v[:, 0] > 0.9 * 110][:, 2].min()
        self.assertLess(bow_keel_z(bow_down), bow_keel_z(level))      # нос погрузился

    def test_cb_fit(self):
        for cb in (0.60, 0.70, 0.80, 0.90):
            h = hull_mod.build_hull(110, 16.5, 5.0, 3.5, 3.4, cb)
            self.assertAlmostEqual(h.cb_model, cb, delta=0.01, msg=f"Cb={cb}")
        # монотонность: чем полнее Cb, тем шире параллельная вставка
        pms = [hull_mod.build_hull(110, 16.5, 5.0, 3.5, 3.4, cb).pm for cb in (0.6, 0.7, 0.8, 0.9)]
        self.assertEqual(pms, sorted(pms))

    def test_unreachable_cb_reports_note(self):
        h = hull_mod.build_hull(110, 16.5, 5.0, 3.5, 3.4, 1.0)
        self.assertTrue(h.notes)
        self.assertLess(h.cb_model, 1.0)

    def test_mesh_integrity(self):
        h = hull_mod.build_hull(110, 16.5, 5.0, 3.5, 3.4, 0.85)
        self.assertTrue(np.isfinite(h.verts).all())
        for name, tri in h.groups.items():
            self.assertGreater(len(tri), 0, name)
            self.assertGreaterEqual(tri.min(), 0, name)
            self.assertLess(tri.max(), len(h.verts), name)

    def test_water_flat_and_wave(self):
        edge = hull_mod.build_hull(110, 16.5, 5.0, 3.5, 3.5, 0.85).wl_edge()
        v, t = hull_mod.build_water(110, 3.5, 3.0, hull_mod.WAVE_NONE, 80, 30, 5, edge)
        self.assertAlmostEqual(v[:, 2].min(), 3.5)
        self.assertAlmostEqual(v[:, 2].max(), 3.5)
        v, _ = hull_mod.build_water(110, 3.5, 3.0, hull_mod.WAVE_HOG, 80, 30, 5, edge)
        self.assertAlmostEqual(v[:, 2].max(), 3.5 + 1.5, places=3)
        self.assertAlmostEqual(v[:, 2].min(), 3.5 - 1.5, places=3)
        mid = v[np.abs(v[:, 0] - 55.0) < 1e-6]
        self.assertTrue(len(mid) and np.allclose(mid[:, 2], 5.0))          # гребень на миделе
        v, _ = hull_mod.build_water(110, 3.5, 3.0, hull_mod.WAVE_SAG, 80, 30, 5, edge)
        mid = v[np.abs(v[:, 0] - 55.0) < 1e-6]
        self.assertTrue(len(mid) and np.allclose(mid[:, 2], 2.0))          # подошва на миделе

    def test_water_does_not_enter_hull(self):
        h = hull_mod.build_hull(110, 16.5, 5.0, 3.5, 3.5, 0.85)
        v, t = hull_mod.build_water(110, 3.5, 0.0, hull_mod.WAVE_NONE, 80, 30, 5, h.wl_edge())
        xs, hw = h.wl_edge()
        c = v[t].mean(axis=1)                                     # центры треугольников воды
        inside = np.abs(c[:, 1]) < np.interp(c[:, 0], xs, hw, left=0, right=0) - 1e-6
        self.assertFalse(inside.any())


# --------------------------------------------------------------------------
class TestSteps(unittest.TestCase):
    CASES = [dict(), dict(Tn=2.2), dict(Tn=1.0), dict(I=0.001, D=50000.0),
             dict(basin=BC.R, V0_kmh=0.0), dict(basin=BC.M, ship_type=ST.TUG),
             dict(basin=BC.L, ship_type=ST.PASSENGER), dict(D=48000.0)]

    def _all(self):
        for kw in self.CASES:
            res = core.calculate(sample(**kw))
            yield kw, res, steps.build_solution(res)

    def test_latex_is_well_formed(self):
        for kw, res, secs in self._all():
            for sec in secs:
                for s in sec.steps:
                    for tex in (s.general, s.calc):
                        self.assertEqual(steps.latex_problems(tex), [], f"{kw}: {s.name}: {tex}")

    def test_values_match_core(self):
        res = core.calculate(sample())
        by_name = {s.name: s for sec in steps.build_solution(res) for s in sec.steps}
        self.assertAlmostEqual(by_name["Волновой изгибающий момент"].value, res.Mv)
        self.assertAlmostEqual(by_name["Коэффициент влияния вибрации"].value, res.kp)
        self.assertAlmostEqual(by_name["Ударный изгибающий момент"].value, res.My)
        self.assertAlmostEqual(by_name["Дополнительный волновой изгибающий момент"].value, res.Mdv)
        self.assertAlmostEqual(by_name["Дополнительная волновая перерезывающая сила"].value, res.Ndv)

    def test_result_printed_in_formula(self):
        res = core.calculate(sample())
        by_name = {s.name: s for sec in steps.build_solution(res) for s in sec.steps}
        self.assertIn(steps.tv(res.Mdv), by_name["Дополнительный волновой изгибающий момент"].calc)
        self.assertIn(steps.tn(res.kp, 4), by_name["Коэффициент влияния вибрации"].calc)

    def test_table_checks_present_and_close(self):
        res = core.calculate(sample())
        by_name = {s.name: s for sec in steps.build_solution(res) for s in sec.steps}
        for key in ("Коэффициент kδ", "Коэффициент kт", "Коэффициент kв"):
            self.assertIn("табл.", by_name[key].check)
            self.assertIn("разница", by_name[key].check)

    def test_phi1_branches_described(self):
        for tn_, token in ((1.0, r"\varphi_1 = 1"), (2.2, r"3-"), (3.4, r"\varphi_1 = 0")):
            res = core.calculate(sample(Tn=tn_))
            st = next(s for sec in steps.build_solution(res) for s in sec.steps if s.name == "Коэффициент φ₁")
            self.assertIn(token, st.calc, f"Tн={tn_}")

    def test_psi_branch_switch(self):
        low = core.calculate(sample())                              # ωк/σ < 1
        names = [s.ref for sec in steps.build_solution(low) for s in sec.steps if s.name == "Коэффициент ψ"]
        self.assertTrue(names and "2.2.10-13" in names[0])
        # ωк/σ между 1,0 и 1,4: подбираем I, чтобы попасть в диапазон
        for I in np.linspace(0.3, 3.0, 60):
            res = core.calculate(sample(I=float(I), D=52000.0))
            if res.psi is not None and 1.0 < res.ratio <= 1.4:
                refs = [s.ref for sec in steps.build_solution(res) for s in sec.steps if s.name == "Коэффициент ψ"]
                self.assertIn("2.2.10-14", refs[0])
                break
        else:
            self.fail("не нашлось I с ωк/σ в диапазоне 1,0…1,4")

    def test_out_of_range_has_warning_and_no_total(self):
        res = core.calculate(sample(I=0.001, D=50000.0))
        secs = steps.build_solution(res)
        vib = next(s for s in secs if s.title.startswith("Влияние"))
        self.assertIn("1,4", vib.warning)
        total = next(s for s in secs if s.title == "Итог")
        self.assertEqual(total.steps, [])
        self.assertTrue(total.warning)

    def test_markdown_export(self):
        res = core.calculate(sample())
        md = steps.solution_markdown(steps.build_solution(res))
        self.assertTrue(md.startswith("# Ход решения"))
        self.assertIn("$$", md)
        self.assertEqual(md.count("$$") % 2, 0)

    def test_cyr_text_wrapping(self):
        self.assertEqual(steps.cyr_text("k_в"), r"k_{\text{в}}")
        self.assertEqual(steps.cyr_text("v_{тв}"), r"v_{\text{тв}}")
        self.assertEqual(steps.cyr_text(r"T_н^0"), r"T_{\text{н}}^0")
        self.assertEqual(steps.cyr_text(r"\text{м}"), r"\text{м}")

    def test_latex_checker_detects_errors(self):
        self.assertTrue(steps.latex_problems(r"\dfrac{1}{2"))
        self.assertTrue(steps.latex_problems(r"\left( x"))
        self.assertTrue(steps.latex_problems("k_в"))
        self.assertEqual(steps.latex_problems(r"\left\{ x \right\}"), [])

    def test_number_format(self):
        self.assertEqual(steps.tn(1234.5, 1), r"1\,234{,}5")
        self.assertEqual(steps.tn(0.85, 3), "0{,}850")
        self.assertEqual(steps.tv(67531.2), r"67\,531")


# --------------------------------------------------------------------------
class TestViewer(unittest.TestCase):
    def _html(self, **kw):
        h = hull_mod.build_hull(110, 16.5, 5.0, 3.5, 3.4, 0.85)
        sc = view3d.build_scene(h, **kw)
        return h, sc, view3d.build_viewer_html(sc, view3d.info_lines(h, "none", 0.0), 520)

    def _payload(self, html: str) -> dict:
        m = re.search(r"const D = (\{.*?\});\n", html, re.S)
        self.assertIsNotNone(m)
        return json.loads(m.group(1).replace("<\\/", "</"))

    def test_page_is_self_contained(self):
        _, _, html = self._html()
        self.assertNotIn("http://", html.replace("http://www.w3.org", ""))
        self.assertNotIn("https://", html)
        self.assertNotIn("<script src", html)

    def test_payload_consistency(self):
        h, sc, html = self._html()
        d = self._payload(html)
        nverts = len(d["v"]) // 3
        for g in d["g"]:
            self.assertEqual(len(g["t"]) % 3, 0)
            self.assertLess(max(g["t"]), nverts)
        self.assertEqual(len(d["dims"]), 4)
        self.assertIn("L = 110,0 м", [x["t"] for x in d["dims"]])
        self.assertIn("T = 3,50 м", [x["t"] for x in d["dims"]])

    def test_z_stretch_scales_heights_only(self):
        _, a, _ = self._html(z_stretch=1.0)
        _, b, _ = self._html(z_stretch=3.0)
        va, vb = np.array(a["v"]).reshape(-1, 3), np.array(b["v"]).reshape(-1, 3)
        self.assertTrue(np.allclose(va[:, :2], vb[:, :2]))
        self.assertTrue(np.allclose(va[:, 2] * 3.0, vb[:, 2], atol=1e-2))
        # подписи показывают истинные размеры
        self.assertEqual([x["t"] for x in a["dims"]], [x["t"] for x in b["dims"]])

    def test_fixed_vs_auto_scale(self):
        _, fx, _ = self._html(fixed_scale=True)
        _, au, _ = self._html(fixed_scale=False)
        self.assertGreater(fx["R"], au["R"])

    def test_script_injection_safe(self):
        h = hull_mod.build_hull(110, 16.5, 5.0, 3.5, 3.4, 0.85)
        sc = view3d.build_scene(h)
        html = view3d.build_viewer_html(sc, ["</script><script>alert(1)</script>"], 520)
        self.assertEqual(html.count("</script>"), 1)


if __name__ == "__main__":
    unittest.main()
