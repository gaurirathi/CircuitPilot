import json
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from circuitpilot.cli import main
from circuitpilot.ir import ModelSpec
from circuitpilot.repair import RepairProposal, run_repair_loop
from circuitpilot.verification import verify_simulation


class DemoRepairPlanner:
    def __init__(self, proposal):
        self.proposal = proposal
        self.calls = 0

    def propose_repair(self, spec, diagnosis):
        self.calls += 1
        self.diagnosis = diagnosis
        return self.proposal


class PidRepairDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo_path = Path(__file__).parents[1] / "examples/closed_loop_pid_repair_demo.json"
        cls.demo_data = json.loads(cls.demo_path.read_text(encoding="utf-8"))
        cls.spec = ModelSpec.model_validate(cls.demo_data)

    def test_demo_loads_exact_existing_verification_criteria(self):
        self.assertEqual(self.spec.verification.max_tracking_error, 0.01)
        self.assertEqual(self.spec.verification.max_overshoot_percent, 5)
        self.assertEqual(self.spec.verification.max_settling_time, 5)
        original = json.loads((Path(__file__).parents[1] / "examples/closed_loop_pid.json").read_text())
        demo_without_criteria = dict(self.demo_data)
        demo_without_criteria.pop("verification")
        self.assertEqual(demo_without_criteria, original)

    def test_measured_overshoot_and_settling_above_limits_fail(self):
        report = verify_simulation(self.spec, {
            "simulation_status": "succeeded",
            "time": list(range(9)),
            "output": [0, 1.10, 1.08, 1.06, 1.04, 1.03, 1.015, 1.0, 1.0],
        })
        overshoot = report.checks["peak_overshoot"]
        settling = report.checks["settling_time"]
        self.assertEqual(overshoot.status, "fail")
        self.assertAlmostEqual(overshoot.value, 10.0)
        self.assertEqual(settling.status, "fail")
        self.assertAlmostEqual(settling.value, 6.0)

    def test_cli_loads_demo_limits_and_reports_failed_measured_metrics(self):
        output = io.StringIO()
        measured = {
            "simulation_status": "succeeded",
            "time": list(range(9)),
            "output": [0, 1.10, 1.08, 1.06, 1.04, 1.03, 1.015, 1.0, 1.0],
        }
        with tempfile.TemporaryDirectory() as temp:
            with patch("sys.argv", [
                "circuitpilot", str(self.demo_path), "--output", temp,
            ]), patch.dict(os.environ, {"CIRCUITPILOT_PROVIDER": "openai"}), \
                 patch("circuitpilot.cli.generate_and_simulate", return_value=measured), \
                 redirect_stdout(output):
                main()
        rendered = json.loads(output.getvalue())
        self.assertEqual(rendered["verification"]["status"], "fail")
        self.assertEqual(rendered["verification"]["checks"]["peak_overshoot"]["status"], "fail")
        self.assertEqual(rendered["verification"]["checks"]["settling_time"]["status"], "fail")

    def test_failed_demo_criterion_triggers_bounded_repair_and_keeps_criteria(self):
        proposal = RepairProposal(
            reason="Reduce overshoot and settling time while preserving the stated limits.",
            changes_summary="Increase the existing controller proportional gain.",
            parameter_updates={"controller": {"P": 3}},
        )
        planner = DemoRepairPlanner(proposal)
        outputs = iter([
            {"simulation_status": "succeeded", "time": list(range(9)),
             "output": [0, 1.10, 1.08, 1.06, 1.04, 1.03, 1.015, 1.0, 1.0]},
            {"simulation_status": "succeeded", "time": [0, 1, 2, 3, 4, 5, 10],
             "output": [0, 0.9, 1.03, 1.01, 1.0, 1.0, 1.0]},
        ])
        generated = []

        def generator(spec, output_dir):
            generated.append(spec)
            result = next(outputs)
            result["model_path"] = str(Path(output_dir) / f"{spec.name}.slx")
            return result

        with tempfile.TemporaryDirectory() as temp:
            report = run_repair_loop(
                self.spec, planner, generator, temp,
                max_repair_attempts=2, run_id="pid-demo-test",
            )
        self.assertEqual(report.status, "pass")
        self.assertEqual([attempt.status for attempt in report.attempts], ["fail", "pass"])
        self.assertEqual(planner.calls, 1)
        self.assertIn("peak_overshoot", {check.name for check in planner.diagnosis.failed_checks})
        self.assertIn("settling_time", {check.name for check in planner.diagnosis.failed_checks})
        self.assertEqual(generated[0].verification, self.spec.verification)
        self.assertEqual(generated[1].verification, self.spec.verification)
        self.assertEqual(generated[1].components[2].parameters["P"], 3)

    def test_proposal_cannot_supply_acceptance_criteria(self):
        with self.assertRaises(Exception):
            RepairProposal.model_validate({
                "reason": "Change limits", "changes_summary": "Weaken limit",
                "parameter_updates": {"controller": {"P": 3}},
                "verification": {"max_overshoot_percent": 100},
            })


if __name__ == "__main__":
    unittest.main()
