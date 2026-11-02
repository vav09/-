"""
Тесты расчётного ядра.

Запуск (любой из вариантов):
    python -m unittest -v
    pytest -v
"""
import math
import unittest

import numpy as np

import pkps_core as core
from pkps_core import BasinClass as BC, ShipType as ST

ORDER = [BC.M, BC.O, BC.R, BC.L]

# Табл. 2.2.10-2 — kδ(δ)
TABLE_KD = {0.55: 0.487, 0.60: 0.527, 0.65: 0.571, 0.70: 0.619, 0.75: 0.670,
            0.80: 0.726, 0.85: 0.787, 0.90: 0.852, 0.95: 0.923, 1.00: 1.000}

# Табл. 2.2.10-3 — kт(a) для «М», «О», «Р», «Л»
TABLE_KT = {
    0.0: (1.000, 1.000, 1.000, 1.000), 0.5: (0.939, 0.897, 0.834, 0.696),
    1.0: (0.881, 0.805, 0.696, 0.485), 1.5: (0.827, 0.722, 0.581, 0.337),
    2.0: (0.776, 0.647, 0.485, 0.235), 2.5: (0.729, 0.581, 0.404, 0.163),
    3.0: (0.684, 0.521, 0.337, 0.114), 3.5: (0.642, 0.467, 0.281, 0.079),
    4.0: (0.603, 0.419, 0.235, 0.055), 4.5: (0.566, 0.376, 0.196, 0.038),
}

# Табл. 2.2.10-4 — kв(b) для «М», «О», «Р», «Л»
TABLE_KB = {
    3.0: (0.911, 0.853, 0.771, 0.610), 6.0: (0.832, 0.735, 0.610, 0.408),
    9.0: (0.762, 0.638, 0.493, 0.295), 12.0: (0.700, 0.559, 0.408, 0.227),
    15.0: (0.646, 0.493, 0.344, 0.183), 18.0: (0.597, 0.439, 0.295, 0.153),
    21.0: (0.553, 0.393, 0.257, 0.131), 24.0: (0.514, 0.355, 0.227, 0.115),
    27.0: (0.479, 0.323, 0.203, 0.102), 30.0: (0.448, 0.295, 0.183, 0.092),
}

TOL = 7e-4   # в таблицах Правил три знака после запятой


def sample_input(**kw) -> core.ShipInput:
    base = dict(basin=BC.O, ship_type=ST.CARGO, L=110.0, B=16.5, T=3.5, Tn=3.4,
                Cb=0.85, V0_kmh=18.0, I=5.0, D=None)
    base.update(kw)
    return core.ShipInput(**base)


class TestAgainstRuleTables(unittest.TestCase):
    def test_table_2_2_10_1_constants(self):
        self.assertEqual(core.CLASS_PARAMS[BC.M].eps, 0.920)
        self.assertEqual(core.CLASS_PARAMS[BC.O].v1, 4.14)
        self.assertEqual(core.CLASS_PARAMS[BC.R].omega_sr, 2.22)
        self.assertEqual(core.CLASS_PARAMS[BC.L].eta, 0.874)

    def test_wave_heights_table_2_1_2(self):
        got = [core.CLASS_PARAMS[c].h for c in ORDER]
        self.assertEqual(got, [3.0, 2.0, 1.2, 0.6])

    def test_k_delta_table(self):
        for cb, expected in TABLE_KD.items():
            self.assertAlmostEqual(core.k_delta(cb), expected, delta=TOL, msg=f"δ={cb}")

    def test_k_t_table(self):
        # kт зависит только от a = 3Tδ/(2δ+1); при δ = 1 получаем a = T.
        for a, row in TABLE_KT.items():
            for cls, expected in zip(ORDER, row):
                p = core.CLASS_PARAMS[cls]
                got = core.k_t(T=a, cb=1.0, eta=p.eta, h=p.h)
                self.assertAlmostEqual(got, expected, delta=TOL, msg=f"{cls.value}, a={a}")

    def test_k_b_table(self):
        # kв зависит только от b = δB; при δ = 1 получаем b = B.
        for b, row in TABLE_KB.items():
            for cls, expected in zip(ORDER, row):
                p = core.CLASS_PARAMS[cls]
                got = core.k_b(B=b, cb=1.0, eta=p.eta, h=p.h)
                self.assertAlmostEqual(got, expected, delta=TOL, msg=f"{cls.value}, b={b}")

    def test_a_b_parameters(self):
        res = core.calculate(sample_input())
        self.assertAlmostEqual(res.a, 3 * 3.5 * 0.85 / (2 * 0.85 + 1), places=9)
        self.assertAlmostEqual(res.b, 0.85 * 16.5, places=9)


class TestFormulaLogic(unittest.TestCase):
    def test_psi_limits_and_continuity(self):
        self.assertAlmostEqual(core.psi_coefficient(0.0), 1.0)
        left = core.psi_coefficient(1.0)
        right = 2.167 * 1.0 - 1.9
        self.assertAlmostEqual(left, right, delta=1e-3)   # кривая непрерывна в точке 1,0
        self.assertAlmostEqual(core.psi_coefficient(1.4), 2.167 * 1.4 - 1.9)

    def test_psi_outside_rules(self):
        with self.assertRaises(core.RulesRangeError):
            core.psi_coefficient(1.41)

    def test_k_mu_never_negative(self):
        self.assertEqual(core.k_mu(50.0), 0.0)
        self.assertAlmostEqual(core.k_mu(0.0), 0.0612)

    def test_k_p_without_vibration_limit(self):
        # ωк → 0: kр → ψ·(1 + 0) = 1
        self.assertAlmostEqual(core.k_p(1.0, 0.0, 0.05), 1.0)

    def test_k_p_resonance_guard(self):
        with self.assertRaises(core.RulesRangeError):
            core.k_p(0.3, 1.0, 0.0)

    def test_phi1_branches(self):
        self.assertEqual(core.phi_1(1.0, 1.0), 1.0)
        self.assertAlmostEqual(core.phi_1(1.25, 1.0), 0.5)
        self.assertEqual(core.phi_1(1.5, 1.0), 0.0)
        self.assertEqual(core.phi_1(2.0, 1.0), 0.0)

    def test_phi0_at_zero(self):
        self.assertEqual(core.phi_0(0.0), 1.0)

    def test_speed_units(self):
        self.assertAlmostEqual(core.speed_to_kmh(10.0, "м/с"), 36.0)
        self.assertAlmostEqual(core.speed_to_kmh(10.0, "уз"), 18.52)
        self.assertAlmostEqual(core.speed_to_kmh(10.0, "км/ч"), 10.0)

    def test_displacement_estimate(self):
        d = core.displacement_from_cb(0.85, 110, 16.5, 3.5, rho=1.0)
        self.assertAlmostEqual(d, 9.81 * 0.85 * 110 * 16.5 * 3.5, places=6)


class TestCalculate(unittest.TestCase):
    def test_assembly_of_total(self):
        r = core.calculate(sample_input())
        self.assertIsNotNone(r.Mdv)
        self.assertAlmostEqual(r.Mdv, r.kp * r.Mv + r.My, places=6)
        self.assertAlmostEqual(r.Ndv, 4 * r.Mdv / r.inp.L, places=6)

    def test_Mv_formula(self):
        r = core.calculate(sample_input())
        p = core.CLASS_PARAMS[BC.O]
        expected = 0.255 * p.eps * r.k_delta * r.k_t * r.k_b * 16.5 * 110 ** 2 * p.h
        self.assertAlmostEqual(r.Mv, expected, places=6)

    def test_no_shock_when_bow_draft_large(self):
        r = core.calculate(sample_input(Tn=5.0))
        self.assertEqual(r.phi1, 0.0)
        self.assertEqual(r.My, 0.0)
        self.assertAlmostEqual(r.Mdv, r.kp * r.Mv, places=6)

    def test_shock_when_bow_draft_small(self):
        r = core.calculate(sample_input(Tn=0.5))
        self.assertEqual(r.phi1, 1.0)
        self.assertGreater(r.My, 0.0)

    def test_every_class_runs(self):
        for cls in ORDER:
            for st in ST:
                r = core.calculate(sample_input(basin=cls, ship_type=st))
                self.assertGreater(r.Mv, 0.0)

    def test_out_of_range_psi_gives_warning_not_crash(self):
        # очень «лёгкий» корпус → малая σ → ωк/σ > 1,4
        r = core.calculate(sample_input(I=0.001, D=50000.0))
        self.assertIsNone(r.psi)
        self.assertIsNone(r.Mdv)
        self.assertTrue(any("ψ" in w for w in r.warnings))
        self.assertGreater(r.Mv, 0.0)       # Mв от kр не зависит

    def test_wave_moment_grows_with_class_severity(self):
        mv = [core.calculate(sample_input(basin=c)).Mv for c in ORDER]
        self.assertGreater(mv[0], mv[1])    # «М» > «О»
        self.assertGreater(mv[1], mv[2])    # «О» > «Р»
        self.assertGreater(mv[2], mv[3])    # «Р» > «Л»

    def test_steps_protocol_not_empty(self):
        r = core.calculate(sample_input())
        symbols = [s.symbol for s in r.steps]
        for needed in ("Mв", "kр", "Mу", "Mдв", "Nдв"):
            self.assertIn(needed, symbols)


class TestValidation(unittest.TestCase):
    def test_negative_or_zero(self):
        for field in ("L", "B", "T", "Tn", "Cb", "I"):
            with self.assertRaises(core.InputError, msg=field):
                core.calculate(sample_input(**{field: 0.0}))

    def test_cb_above_one(self):
        with self.assertRaises(core.InputError):
            core.calculate(sample_input(Cb=1.2))

    def test_negative_speed(self):
        with self.assertRaises(core.InputError):
            core.calculate(sample_input(V0_kmh=-1.0))

    def test_auto_D_warning(self):
        r = core.calculate(sample_input(D=None))
        self.assertTrue(any("оценено" in w for w in r.warnings))
        r2 = core.calculate(sample_input(D=50000.0))
        self.assertFalse(any("оценено" in w for w in r2.warnings))


class TestProfiles(unittest.TestCase):
    def test_moment_profile_shape(self):
        x, m = core.moment_profile(1000.0, plateau=0.5)
        self.assertAlmostEqual(m[0], 0.0)
        self.assertAlmostEqual(m[-1], 0.0)
        self.assertAlmostEqual(float(m.max()), 1000.0)
        self.assertAlmostEqual(float(np.interp(0.5, x, m)), 1000.0)
        self.assertAlmostEqual(float(np.interp(0.25, x, m)), 1000.0)   # начало полки
        self.assertAlmostEqual(float(np.interp(0.125, x, m)), 500.0)   # середина наклона

    def test_moment_profile_plateau_030(self):
        x, m = core.moment_profile(1000.0, plateau=0.3)
        self.assertAlmostEqual(float(np.interp(0.35, x, m)), 1000.0)
        self.assertAlmostEqual(float(np.interp(0.65, x, m)), 1000.0)

    def test_moment_profile_bad_plateau(self):
        with self.assertRaises(core.InputError):
            core.moment_profile(1.0, plateau=1.0)

    def test_shear_profile_fig_2_2_10(self):
        x, n = core.shear_profile(100.0)
        self.assertAlmostEqual(float(np.interp(0.15, x, n)), 100.0)
        self.assertAlmostEqual(float(np.interp(0.30, x, n)), 100.0)
        self.assertAlmostEqual(float(np.interp(0.50, x, n)), 70.0)    # 0,7·Nдв в середине
        self.assertAlmostEqual(float(np.interp(0.85, x, n)), 100.0)
        self.assertAlmostEqual(float(n[0]), 0.0)
        self.assertAlmostEqual(float(n[-1]), 0.0)

    def test_shear_slope_matches_formula(self):
        # Наклон Mдв на оконечности = Mдв/(0,25L) = 4Mдв/L = Nдв
        L = 100.0
        Mdv = 5000.0
        x, m = core.moment_profile(Mdv, plateau=0.5)
        slope = (np.interp(0.1, x, m) - np.interp(0.0, x, m)) / (0.1 * L)
        self.assertAlmostEqual(slope, 4 * Mdv / L, places=6)

    def test_combine_with_still_water(self):
        x, m = core.moment_profile(1000.0, plateau=0.5)
        mtv, plus, minus = core.combine_with_still_water([0, 0.5, 1], [0, 2000, 0], x, m)
        i = int(np.argmin(abs(x - 0.5)))
        self.assertAlmostEqual(mtv[i], 2000.0)
        self.assertAlmostEqual(plus[i], 3000.0)
        self.assertAlmostEqual(minus[i], 1000.0)

    def test_combine_needs_two_points(self):
        x, m = core.moment_profile(1000.0)
        with self.assertRaises(core.InputError):
            core.combine_with_still_water([0.5], [100], x, m)


class TestFormat(unittest.TestCase):
    def test_format_value(self):
        self.assertEqual(core.format_value(None), "—")
        self.assertEqual(core.format_value(0.0), "0")
        self.assertEqual(core.format_value(12345.6), "12\u00a0346")
        self.assertEqual(core.format_value(0.5), "0.5000")


if __name__ == "__main__":
    unittest.main()
