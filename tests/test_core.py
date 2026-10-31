"""
Unit tests for the core pieces of TrustFlow. Standard library only:

    .venv/bin/python -m unittest discover tests

No dataset is needed; the simulator generates what the integration tests use.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import drift_guard as G                      # noqa: E402
import energy_check as E                     # noqa: E402
from alarms_per_day import runs, count       # noqa: E402
from swat_invariants import persist          # noqa: E402
from swro_sim import simulate, hp_power, P_SUC   # noqa: E402
from invariants_swro import fit, violations  # noqa: E402


class Persistence(unittest.TestCase):
    def test_short_runs_dropped_long_runs_kept_whole(self):
        flag = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 1], bool)
        out = persist(flag, 3)
        np.testing.assert_array_equal(out, [0, 0, 0, 0, 1, 1, 1, 1, 0, 0])

    def test_k1_is_identity(self):
        flag = np.array([1, 0, 1], bool)
        np.testing.assert_array_equal(persist(flag, 1), flag)


class AlarmEpisodes(unittest.TestCase):
    def test_runs(self):
        self.assertEqual(runs([0, 1, 1, 0, 1]), [(1, 3), (4, 5)])
        self.assertEqual(runs([0, 0]), [])

    def test_count_does_not_bridge_excluded_rows(self):
        flag = np.ones(6, bool)
        mask = np.array([1, 1, 0, 1, 1, 1], bool)
        r = count(flag, 1, mask)
        self.assertEqual(r["episodes"], 2)
        self.assertEqual(r["flagged_rows"], 5)

    def test_merge_window(self):
        flag = np.zeros(1000, bool)
        flag[[10, 100, 900]] = True          # 90 s apart, then 800 s apart
        r = count(flag, 1)
        self.assertEqual(r["episodes"], 3)
        self.assertEqual(r["merged_alarms"], 2)
        self.assertAlmostEqual(r["days"], 1000 / 86400, delta=0.0005)   # reported to 3 decimals


class RateCappedReference(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(200, dtype=float)

    def test_follows_slow_change_exactly(self):
        ref = 0.001 * self.t
        np.testing.assert_allclose(G.cap(ref, self.t, rate=0.01), ref)

    def test_limits_a_step(self):
        ref = np.where(self.t < 100, 0.0, 1.0)
        out = G.cap(ref, self.t, rate=0.01)
        self.assertAlmostEqual(out[100], 0.01)
        self.assertAlmostEqual(out[149], 0.50)
        self.assertTrue(np.all(np.diff(out) <= 0.01 + 1e-12))

    def test_reset_jumps_to_raw(self):
        ref = np.where(self.t < 100, 0.0, 1.0)
        out = G.cap(ref, self.t, rate=0.01, resets=[100])
        self.assertEqual(out[100], 1.0)

    def test_fit_cap_on_a_ramp(self):
        ref = 0.002 * self.t                  # 0.002 per minute
        self.assertAlmostEqual(G.fit_cap(ref, self.t), G.MARGIN * 0.002, places=9)

    def test_hold_freezes_while_out_of_tolerance(self):
        ref = np.where(self.t < 100, 0.0, 1.0)
        q = ref.copy()                        # the checked quantity jumps with it
        out = G.cap(ref, self.t, rate=0.01, q=q, tol=0.1)
        self.assertTrue(np.all(out[100:] == 0.0))

    def test_hold_releases_when_back_in_tolerance(self):
        ref = np.where(self.t < 100, 0.0, 1.0)
        q = np.where((self.t >= 100) & (self.t < 120), 1.0, 0.0)   # excursion ends at 120
        out = G.cap(ref, self.t, rate=0.01, q=q, tol=0.1)
        self.assertTrue(np.all(out[100:120] == 0.0))
        self.assertGreater(out[150], 0.0)

    def test_without_hold_unchanged(self):
        ref = np.sin(self.t / 20)
        np.testing.assert_array_equal(G.cap(ref, self.t, 0.02), G.cap(ref, self.t, 0.02, q=None, tol=None))


class EnergyCheck(unittest.TestCase):
    def test_residual_is_minus_log_efficiency(self):
        import pandas as pd
        Q, P, eta = np.array([900.0, 1000.0]), np.array([60.0, 62.0]), 0.75
        W = Q * (P - P_SUC) / 36.0 / eta
        r = E.residual(pd.DataFrame({"Q_feed": Q, "P_feed": P, "W_hp": W}))
        np.testing.assert_allclose(r, -np.log(eta))

    def test_simulator_power_consistent_with_check(self):
        Q, P = 1000.0, 60.0
        W = hp_power(Q, P, day=0, days=60)
        eta = Q * (P - P_SUC) / 36.0 / W
        self.assertTrue(0.6 < eta < 0.9)


class SimulatorAndChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.truth, cls.rep, cls.ev = simulate(days=4, seed=0)

    def test_no_false_alarm_on_calibration_rows(self):
        cal = fit(self.rep)                   # tolerance = 1.2 x worst deviation, by construction
        self.assertFalse(violations(self.rep, cal).any())

    def test_spoofed_flow_breaks_flow_balance(self):
        cal = fit(self.rep)
        x = self.rep.copy()
        x["Q_feed"] = x["Q_feed"] * 1.2
        self.assertGreater(violations(x, cal)[:, 0].mean(), 0.9)


if __name__ == "__main__":
    unittest.main()
