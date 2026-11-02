"""Тесты эквивалентного бруса, проектов и отчётов."""
import json
import math
import unittest

import pkps_beam as bm
import pkps_core as core
import pkps_project as pj
import pkps_report as rp
import pkps_steps as steps
from pkps_core import BasinClass, ShipType


def box(t_deck=10.0, flexible=False):
    """Коробка: палуба и днище 1000×10 мм на расстоянии 2 м, борта 2000×10 мм (полное сечение)."""
    return [
        bm.Element("палуба", bm.T_HPLATE, b=1000, t=t_deck, z=2.0, flexible=flexible, a=500, b_panel=2000),
        bm.Element("днище", bm.T_HPLATE, b=1000, t=10, z=0.0),
        bm.Element("борт", bm.T_VPLATE, n=2, b=2000, t=10, z=1.0),
    ]


class BeamGeometry(unittest.TestCase):
    def test_box_props(self):
        sp = bm.section_props(box(), half=False)
        self.assertAlmostEqual(sp.A, 0.01 + 0.01 + 0.04, places=9)
        self.assertAlmostEqual(sp.z0, 1.0, places=9)
        I = 2 * (0.01 * 1.0 ** 2 + 1.0 * 0.01 ** 3 / 12) + 2 * 0.01 * 2.0 ** 3 / 12
        self.assertAlmostEqual(sp.I, I, places=9)
        self.assertAlmostEqual(sp.W_deck, I / (2.005 - 1.0), places=9)

    def test_half_doubles(self):
        full = bm.section_props(box(), half=False)
        half = bm.section_props(box(), half=True)
        self.assertAlmostEqual(half.I, 2 * full.I)
        self.assertAlmostEqual(half.z0, full.z0)

    def test_profile_up_down(self):
        up = bm.Element("тавр", bm.T_TEE, b=200, t=10, bf=100, tf=10, z=0.0)
        dn = bm.Element("тавр", bm.T_TEE, b=200, t=10, bf=100, tf=10, z=5.0, down=True)
        self.assertAlmostEqual(up.area(), 0.003)
        self.assertAlmostEqual(up.centroid(), (0.002 * 0.1 + 0.001 * 0.205) / 0.003)
        self.assertAlmostEqual(5.0 - dn.centroid(), up.centroid())
        self.assertAlmostEqual(up.own_inertia(), dn.own_inertia())
        lo, hi = up.z_range()
        self.assertAlmostEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.21)

    def test_inclined_limits(self):
        h = bm.Element("", bm.T_IPLATE, b=1000, t=10, theta=0, z=0)
        v = bm.Element("", bm.T_IPLATE, b=1000, t=10, theta=90, z=0)
        self.assertAlmostEqual(h.own_inertia(), 1.0 * 0.01 ** 3 / 12)
        self.assertAlmostEqual(v.own_inertia(), 0.01 * 1.0 ** 3 / 12)

    def test_records_roundtrip(self):
        els = bm.elements_from_records(bm.example_records())
        self.assertEqual(len(els), 10)
        rows = bm.example_records() + [dict(bm.DEFAULT_ROW)]       # пустая строка пропускается
        self.assertEqual(len(bm.elements_from_records(rows)), 10)

    def test_validation(self):
        with self.assertRaises(bm.BeamError):
            bm.calculate_beam([bm.Element("x", bm.T_HPLATE, b=0, t=10)])
        with self.assertRaises(bm.BeamError):
            bm.calculate_beam([bm.Element("x", bm.T_HPLATE, b=100, t=10, flexible=True)])   # нет a
        with self.assertRaises(bm.BeamError):
            bm.calculate_beam([bm.Element("x", bm.T_TEE, b=100, t=10)])                     # нет пояска


class BeamRules(unittest.TestCase):
    def test_k_n(self):
        self.assertAlmostEqual(bm.k_n(235), 1.0)
        q = 355 / 235 - 1
        self.assertAlmostEqual(bm.k_n(355), 1 - 0.089 * q - 0.129 * q * q)
        self.assertAlmostEqual(bm.sigma_0(235), 235)

    def test_allowable(self):
        self.assertEqual(bm.allowable_factor("М", False), 0.70)
        self.assertEqual(bm.allowable_factor("Р", False), 0.75)
        self.assertEqual(bm.allowable_factor("О", True), 0.60)

    def test_sigma_cr_branches(self):
        # σэ = 78,5·(100t/a)²: t=10, a=1000 → 78,5 (≤ 0,6ReH) — упругая область
        self.assertAlmostEqual(bm.sigma_cr_plate(10, 1000, 235), 78.5)
        se = bm.sigma_euler_plate(10, 500)              # 314
        self.assertAlmostEqual(bm.sigma_cr_plate(10, 500, 235), (1.63 - 0.8 * math.sqrt(235 / se)) * 235)
        self.assertAlmostEqual(bm.sigma_cr_plate(20, 300, 235), 235)

    def test_table_2239(self):
        self.assertAlmostEqual(bm.phi_table_2239(6, True), 0.07)
        self.assertAlmostEqual(bm.phi_table_2239(10, False), (0.33 + 0.56) / 2)
        self.assertAlmostEqual(bm.phi_table_2239(3, False), 0.07)       # ниже таблицы — крайнее

    def test_rigid_fraction(self):
        el = bm.Element("", bm.T_HPLATE, b=1000, t=10, a=600, system=bm.SYS_LONG)
        self.assertAlmostEqual(bm.rigid_fraction(el), 0.5)              # 2·0,25a / a
        el = bm.Element("", bm.T_HPLATE, b=1000, t=4, a=600, system=bm.SYS_LONG)
        self.assertAlmostEqual(bm.rigid_fraction(el), 2 * 100 / 600)    # 25t = 100 < 150
        el = bm.Element("", bm.T_HPLATE, b=1000, t=10, a=600, b_panel=3000, system=bm.SYS_TRANS)
        self.assertAlmostEqual(bm.rigid_fraction(el), 2 * 150 / 3000)

    def test_tension_long_not_reduced(self):
        el = bm.Element("", bm.T_HPLATE, b=1000, t=6, a=800, flexible=True)
        self.assertEqual(bm.reduce_plate(el, +100).phi_eff, 1.0)
        red = bm.reduce_plate(el, -100)
        scr = bm.sigma_cr_plate(6, 800, 235)
        self.assertAlmostEqual(red.phi, scr / 100)
        fr = 2 * 150 / 800                                  # 25t = 150 мм < 0,25a = 200 мм
        self.assertAlmostEqual(red.phi_eff, fr + (1 - fr) * scr / 100)

    def test_stress_sign(self):
        r = bm.calculate_beam(box(), False, M_sag=10000, M_hog=10000)
        sag, hog = r.cases
        self.assertLess(sag.final.sigma_deck, 0)       # прогиб — палуба сжата
        self.assertGreater(hog.final.sigma_deck, 0)
        sp = r.zero
        self.assertAlmostEqual(hog.final.sigma_deck, 10000 * (sp.z_top - sp.z0) * 1e-3 / sp.I)

    def test_reduction_lowers_I_and_converges(self):
        els = box(t_deck=5, flexible=True)
        r = bm.calculate_beam(els, False, M_sag=30000, M_hog=None)
        cr = r.cases[0]
        self.assertTrue(cr.converged)
        self.assertLess(cr.final.props.I, r.zero.I)
        self.assertLessEqual(cr.final.diff, bm.CONVERGENCE)
        self.assertLess(cr.final.props.z0, r.zero.z0)   # палуба «полегчала» — НО опустилась

    def test_example_checks(self):
        r = bm.calculate_beam(bm.elements_from_records(bm.example_records()), True, 67531, 67531, "О")
        self.assertEqual(len(r.checks), 20)
        self.assertTrue(all(c.allowable > 0 for c in r.checks))
        self.assertTrue(r.worst.ok)
        r2 = bm.calculate_beam(r.elements, True, 300000, None, "О")
        self.assertFalse(r2.worst.ok)


class Project(unittest.TestCase):
    def test_roundtrip(self):
        vals = dict(pj.DEFAULTS)
        vals.update(L=95.0, basin=BasinClass.L, ship_type=ShipType.TUG, project_name="Буксир")
        raw = pj.dump(vals, bm.example_records(), [{"x/L": 0.5, "Mтв, кН·м": float("nan")}])
        v, beam_rows, mtv = pj.load(raw)
        self.assertEqual(v["L"], 95.0)
        self.assertIs(v["basin"], BasinClass.L)
        self.assertIs(v["ship_type"], ShipType.TUG)
        self.assertEqual(len(beam_rows), 10)
        self.assertIsNone(mtv[0]["Mтв, кН·м"])

    def test_bad_files(self):
        for raw in (b"not json", b'{"format": "other"}', json.dumps(
                {"format": pj.FORMAT, "version": 99}).encode()):
            with self.assertRaises(pj.ProjectError):
                pj.load(raw)
        bad = json.dumps({"format": pj.FORMAT, "version": 1, "inputs": {"basin": "Z"}}).encode()
        with self.assertRaises(pj.ProjectError):
            pj.load(bad)

    def test_missing_fields_default(self):
        v, b, m = pj.load(json.dumps({"format": pj.FORMAT, "version": 1, "inputs": {"L": 50}}).encode())
        self.assertEqual(v["L"], 50.0)
        self.assertEqual(v["B"], pj.DEFAULTS["B"])
        self.assertEqual(b, [])

    def test_apply_editor_state(self):
        rows = [{"a": 1}, {"a": 2}, {"a": 3}]
        st = {"edited_rows": {"0": {"a": 10}}, "deleted_rows": [1], "added_rows": [{"a": 4, "_index": 9}]}
        out = pj.apply_editor_state(rows, st, ["a", "b"], {"b": "x"})
        self.assertEqual(out, [{"a": 10}, {"a": 3}, {"a": 4, "b": "x"}])
        self.assertEqual(rows[0]["a"], 1)               # исходные строки не меняются

    def test_file_name(self):
        self.assertEqual(pj.file_name("Сухогруз 01/А"), "Сухогруз_01_А.pkps.json")


class Report(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        res = core.calculate(core.ShipInput(basin=BasinClass.O, ship_type=ShipType.CARGO, L=110, B=16.5,
                                            T=3.5, Tn=1.6, Cb=0.85, V0_kmh=18, I=5, H=5))
        cls.data = rp.ReportData("Тест", "Проект", [("L, м", "110")], [("Mдв", "1")],
                                 sections=steps.build_solution(res),
                                 protocol=[["a", "b", "c", "d", "e"], ["1", "2", "3", "4", "5"]])

    def test_mathtext_cases(self):
        lines = rp.to_mathtext(r"\varphi = \begin{cases}1, & a\le b\\ 0, & a\ge b\end{cases}")
        self.assertEqual(len(lines), 2)
        self.assertIn(r"\leq", lines[0])
        self.assertIn(r"\geq", lines[1])

    def test_all_formulas_render(self):
        bad = [line for sec in self.data.sections for s in sec.steps for t in (s.general, s.calc) if t
               for png, line in rp.formula_images(t) if png is None]
        self.assertEqual(bad, [])

    def test_pdf_and_docx(self):
        pdf = rp.build_pdf(self.data, list(rp.SECTIONS))
        self.assertTrue(pdf.startswith(b"%PDF"))
        docx = rp.build_docx(self.data, ["inputs", "steps", "protocol"])
        self.assertTrue(docx.startswith(b"PK"))


if __name__ == "__main__":
    unittest.main()
