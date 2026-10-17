import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from circuitpilot.cli import main
from circuitpilot.ir import ModelSpec
from circuitpilot.repair import run_repair_loop
from circuitpilot.run_reports import RunReportWriter


class RunReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_path = Path(__file__).parents[1] / "examples" / "closed_loop_pid.json"
        cls.base_spec = ModelSpec.model_validate_json(cls.source_path.read_text(encoding="utf-8"))

    def spec_with_criteria(self, criteria):
        data = self.base_spec.model_dump(mode="json")
        data["verification"] = criteria
        return ModelSpec.model_validate(data)

    def invoke_cli(self, temp, spec, simulation):
        spec_path = Path(temp) / "input.json"
        spec_path.write_text(spec.model_dump_json(), encoding="utf-8")
        output_root = Path(temp) / "artifacts"
        with patch("sys.argv", ["circuitpilot", str(spec_path), "--output", str(output_root)]), \
             patch("circuitpilot.cli.generate_and_simulate", return_value=simulation):
            main()
        report_dirs = list((output_root / "run_reports").iterdir())
        self.assertEqual(len(report_dirs), 1)
        return report_dirs[0]

    def test_successful_cli_run_writes_metrics_thresholds_and_model_details(self):
        spec = self.spec_with_criteria({
            "max_tracking_error": 0.1,
            "max_overshoot_percent": 20,
            "max_settling_time": 10,
        })
        result = {
            "model_path": "C:/models/ClosedLoopPID.slx",
            "model_opened": True,
            "diagram_update": "succeeded",
            "simulation_status": "succeeded",
            "time": [0, 1, 2, 4, 6, 10],
            "output": [0, 0.8, 1.1, 1.05, 1.01, 1.0],
        }
        with tempfile.TemporaryDirectory() as temp:
            report_dir = self.invoke_cli(temp, spec, result)
            data = json.loads((report_dir / "verification_report.json").read_text(encoding="utf-8"))
            markdown = (report_dir / "verification_report.md").read_text(encoding="utf-8")

        self.assertEqual(data["overall_status"], "pass")
        self.assertEqual(data["model_generation_status"], "succeeded")
        self.assertEqual(data["matlab_startup_status"], "succeeded")
        self.assertEqual(data["diagram_update_status"], "succeeded")
        self.assertEqual(data["simulation_status"], "succeeded")
        self.assertEqual(data["sample_count"], 6)
        self.assertEqual(data["simulation_time_range"], "0 to 10 s")
        self.assertAlmostEqual(data["verification"]["checks"]["peak_overshoot"]["value"], 10.0)
        self.assertEqual(data["verification"]["checks"]["peak_overshoot"]["criterion"], "<= 20.0%")
        self.assertIn("PASS", markdown)
        self.assertIn("10.000000000000009 %", markdown)
        self.assertIn("<= 20.0%", markdown)
        self.assertIn("ClosedLoopPID.slx", markdown)

    def test_verification_failure_is_reported_with_measured_threshold(self):
        spec = self.spec_with_criteria({
            "max_tracking_error": 0.01,
            "max_overshoot_percent": 5,
            "max_settling_time": 5,
        })
        result = {
            "simulation_status": "succeeded",
            "time": [0, 1, 2, 4, 6, 10],
            "output": [0, 0.8, 1.1, 1.05, 1.01, 1.0],
        }
        with tempfile.TemporaryDirectory() as temp:
            report_dir = self.invoke_cli(temp, spec, result)
            report = (report_dir / "verification_report.md").read_text(encoding="utf-8")
        self.assertIn("Overall status: **FAIL**", report)
        self.assertIn("Peak overshoot exceeds the specified limit.", report)
        self.assertIn("<= 5.0%", report)
        self.assertIn("Settling time", report)
        self.assertIn("<= 5.0 s; 2% band", report)
        self.assertIn("adjust the structured model parameters", report)

    def test_handled_simulation_error_is_reported_without_exception_text(self):
        spec = self.spec_with_criteria({"max_tracking_error": 0.01})
        sensitive_message = "authorization=secret-value private payload"
        with tempfile.TemporaryDirectory() as temp:
            spec_path = Path(temp) / "input.json"
            spec_path.write_text(spec.model_dump_json(), encoding="utf-8")
            output_root = Path(temp) / "artifacts"
            with patch("sys.argv", ["circuitpilot", str(spec_path), "--output", str(output_root)]), \
                 patch("circuitpilot.cli.generate_and_simulate", side_effect=RuntimeError(sensitive_message)):
                with self.assertRaisesRegex(RuntimeError, "authorization=secret-value"):
                    main()
            report_dir = next((output_root / "run_reports").iterdir())
            json_text = (report_dir / "verification_report.json").read_text(encoding="utf-8")
            markdown = (report_dir / "verification_report.md").read_text(encoding="utf-8")

        self.assertIn('"overall_status": "error"', json_text)
        self.assertIn('"exception_type": "RuntimeError"', json_text)
        self.assertIn("model generation and simulation", markdown)
        self.assertNotIn(sensitive_message, json_text)
        self.assertNotIn(sensitive_message, markdown)
        self.assertNotIn("secret-value", json_text + markdown)

    def test_partial_run_information_still_produces_diagnostic_report(self):
        with tempfile.TemporaryDirectory() as temp:
            writer = RunReportWriter(
                Path(temp) / "run_reports" / "run-one",
                run_id="run-one",
                input_path="spec.json",
            )
            json_path = writer.directory / "verification_report.json"
            markdown_path = writer.directory / "verification_report.md"
            data = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(data["overall_status"], "inconclusive")
        self.assertIsNone(data["verification"])
        self.assertIn("Verification results are not available yet", markdown)
        self.assertIn("not started", markdown)

    def test_repair_checkpoints_update_markdown_and_final_outcome(self):
        spec = self.spec_with_criteria({
            "max_tracking_error": 0.05,
            "max_overshoot_percent": 15,
            "max_settling_time": 8,
        })
        result = {
            "model_path": "model.slx",
            "model_opened": True,
            "diagram_update": "succeeded",
            "simulation_status": "succeeded",
            "time": [0, 1, 2, 3, 5, 10],
            "output": [0, 0.8, 1.05, 1.02, 1.0, 0.8],
        }
        observed_checkpoint = []

        class TimedOutPlanner:
            def propose_repair(self, current, diagnosis):
                path = Path(temp_root) / "repairs" / "report-test" / "verification_report.md"
                observed_checkpoint.append(path.read_text(encoding="utf-8"))
                raise TimeoutError("sensitive provider response")

        with tempfile.TemporaryDirectory() as temp_root:
            report = run_repair_loop(
                spec,
                TimedOutPlanner(),
                lambda current, output_dir: result,
                temp_root,
                max_repair_attempts=1,
                run_id="report-test",
                input_path=str(self.source_path),
            )

            run_dir = Path(temp_root) / "repairs" / "report-test"
            checkpoint = observed_checkpoint[0]
            final_markdown = (run_dir / "verification_report.md").read_text(encoding="utf-8")
            repair_json = json.loads((run_dir / "repair_report.json").read_text(encoding="utf-8"))

        self.assertIn("Attempt 0 — original", checkpoint)
        self.assertIn("Attempt 1 — repair", checkpoint)
        self.assertIn("Current stage: planning", checkpoint)
        self.assertIn("IN PROGRESS", checkpoint)
        self.assertIn("Overall status: **ERROR**", final_markdown)
        self.assertIn("<= 0.05 output units", final_markdown)
        self.assertIn("Final repair outcome: error", final_markdown)
        self.assertEqual(repair_json["status"], "error")
        self.assertNotIn("sensitive provider response", final_markdown)


if __name__ == "__main__":
    unittest.main()
