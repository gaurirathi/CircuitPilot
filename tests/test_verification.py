import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from circuitpilot.cli import main
from circuitpilot.ir import ModelSpec
from circuitpilot.verification import verify_simulation


class SimulationVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example = json.loads((Path(__file__).parents[1] / "examples/closed_loop_pid.json").read_text())

    def spec(self, verification=None):
        data = dict(self.example)
        if verification is not None:
            data["verification"] = verification
        return ModelSpec.model_validate(data)

    def test_good_response_passes_all_user_defined_criteria(self):
        spec = self.spec({
            "max_tracking_error": 0.01,
            "max_overshoot_percent": 11,
            "max_settling_time": 3,
            "settling_band_percent": 2,
        })
        report = verify_simulation(spec, {
            "simulation_status": "succeeded",
            "time": [0, 0.5, 1, 1.5, 2, 3, 4, 5],
            "output": [0, 0.8, 1.1, 1.05, 1.01, 1, 1, 1],
        })
        self.assertEqual(report.status, "pass")
        self.assertAlmostEqual(report.checks["final_value_tracking_error"].value, 0)
        self.assertAlmostEqual(report.checks["peak_overshoot"].value, 10)
        self.assertAlmostEqual(report.checks["settling_time"].value, 2)

    def test_missing_limits_report_metrics_without_inventing_thresholds(self):
        report = verify_simulation(self.spec(), {
            "simulation_status": "succeeded",
            "time": [0, 1, 2, 3, 4],
            "output": [0, 0.8, 1.1, 1.02, 1],
        })
        self.assertEqual(report.status, "inconclusive")
        for key in ("final_value_tracking_error", "peak_overshoot", "settling_time"):
            self.assertEqual(report.checks[key].status, "inconclusive")
            self.assertIsNotNone(report.checks[key].value)
            self.assertIsNone(report.checks[key].criterion)

    def test_non_finite_response_fails_integrity_check(self):
        report = verify_simulation(self.spec(), {
            "simulation_status": "succeeded",
            "time": [0, 1],
            "output": [0, float("nan")],
        })
        self.assertEqual(report.status, "fail")
        self.assertEqual(report.checks["response_finite"].status, "fail")
        self.assertIn("NaN or infinite", report.checks["response_finite"].reason)

    def test_tracking_error_limit_can_fail(self):
        report = verify_simulation(self.spec({
            "max_tracking_error": 0.05,
        }), {
            "simulation_status": "succeeded",
            "time": [0, 1, 2, 3],
            "output": [0, 0.4, 0.7, 0.8],
        })
        check = report.checks["final_value_tracking_error"]
        self.assertEqual(check.status, "fail")
        self.assertAlmostEqual(check.value, 0.2)
        self.assertEqual(report.status, "fail")

    def test_overshoot_limit_can_fail(self):
        report = verify_simulation(self.spec({
            "max_overshoot_percent": 5,
        }), {
            "simulation_status": "succeeded",
            "time": [0, 1, 2, 3, 4],
            "output": [0, 1.3, 1.1, 1, 1],
        })
        self.assertEqual(report.checks["peak_overshoot"].status, "fail")
        self.assertAlmostEqual(report.checks["peak_overshoot"].value, 30)

    def test_diverging_response_fails_required_settling_criterion(self):
        report = verify_simulation(self.spec({"max_settling_time": 3}), {
            "simulation_status": "succeeded",
            "time": [0, 1, 2, 3, 4],
            "output": [0, 1, 2, 4, 8],
        })
        check = report.checks["settling_time"]
        self.assertEqual(check.status, "fail")
        self.assertIn("did not enter and remain", check.reason)

    def test_insufficient_data_is_inconclusive(self):
        report = verify_simulation(self.spec({"max_settling_time": 2}), {
            "simulation_status": "succeeded",
            "time": [0],
            "output": [0],
        })
        self.assertEqual(report.checks["sample_data"].status, "inconclusive")
        self.assertEqual(report.checks["settling_time"].status, "inconclusive")
        self.assertEqual(report.status, "inconclusive")

    def test_cli_includes_structured_verification_report(self):
        spec = self.spec({"max_tracking_error": 0.1})
        simulation = {
            "simulation_status": "succeeded",
            "time": [0, 1, 2],
            "output": [0, 0.95, 1],
        }
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp:
            spec_path = Path(temp) / "spec.json"
            spec_path.write_text(spec.model_dump_json(), encoding="utf-8")
            with patch("sys.argv", ["circuitpilot", str(spec_path), "--output", str(Path(temp) / "artifacts")]), \
                 patch.dict(os.environ, {"CIRCUITPILOT_PROVIDER": "openai"}), \
                 patch("circuitpilot.cli.generate_and_simulate", return_value=simulation), \
                 redirect_stdout(output):
                main()
        rendered = json.loads(output.getvalue())
        self.assertEqual(rendered["verification"]["status"], "inconclusive")
        self.assertEqual(rendered["verification"]["checks"]["simulation_success"]["status"], "pass")
        self.assertEqual(rendered["verification"]["checks"]["final_value_tracking_error"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
