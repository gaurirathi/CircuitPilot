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
from circuitpilot.repair import RepairProposal, RepairRejected, apply_repair, run_repair_loop
from circuitpilot.planner import GeminiPlanner


class FakeRepairPlanner:
    def __init__(self, proposals):
        self.proposals = list(proposals)
        self.calls = 0

    def propose_repair(self, spec, diagnosis):
        self.calls += 1
        self.last_spec = spec
        self.last_diagnosis = diagnosis
        proposal = self.proposals.pop(0)
        if isinstance(proposal, Exception):
            raise proposal
        return proposal


class RepairLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((Path(__file__).parents[1] / "examples/closed_loop_pid.json").read_text())

    def make_spec(self, criteria=True):
        data = json.loads(json.dumps(self.data))
        if criteria:
            data["verification"] = {
                "max_tracking_error": 0.05,
                "max_overshoot_percent": 15,
                "max_settling_time": 8,
            }
        return ModelSpec.model_validate(data)

    @staticmethod
    def result(final=1.0):
        return {
            "simulation_status": "succeeded",
            "model_path": "simulated.slx",
            "time": [0, 1, 2, 3, 5, 10],
            "output": [0, 0.8, 1.05, 1.02, 1.0, final],
        }

    def run_loop(self, spec, planner, generator, temp, max_attempts=2):
        return run_repair_loop(
            spec, planner, generator, temp,
            max_repair_attempts=max_attempts, run_id="test-run",
        )

    def test_repair_scope_preserves_topology_output_and_acceptance_criteria(self):
        spec = self.make_spec()
        before = spec.model_dump(mode="json")
        repaired = apply_repair(spec, RepairProposal(
            reason="Tune existing controller", changes_summary="Adjust P", parameter_updates={"controller": {"P": 3}},
        ))
        self.assertEqual(repaired.connections, spec.connections)
        self.assertEqual(repaired.output_component, spec.output_component)
        self.assertEqual(repaired.verification, spec.verification)
        self.assertEqual(spec.model_dump(mode="json"), before)
        for update in ({"plant": {"Numerator": [2]}}, {"controller": {"Controller": "P"}}):
            with self.assertRaises(RepairRejected):
                apply_repair(spec, RepairProposal(reason="x", changes_summary="x", parameter_updates=update))

    def test_pid_allowlist_rejects_nonfinite_and_out_of_range_values(self):
        spec = self.make_spec()
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                proposal = RepairProposal(reason="tune", changes_summary="change P", parameter_updates={"controller": {"P": value}})
                with self.assertRaises(RepairRejected):
                    apply_repair(spec, proposal)
        for stop_time in (float("nan"), float("inf"), 0, 86401):
            with self.subTest(stop_time=stop_time):
                with self.assertRaises(Exception):
                    RepairProposal(reason="tune", changes_summary="extend run", stop_time=stop_time)

    def test_original_artifacts_and_report_are_not_overwritten_and_run_ids_do_not_collide(self):
        spec = self.make_spec()
        planner = FakeRepairPlanner([RepairProposal(
            reason="PID adjustment", changes_summary="Increase P", parameter_updates={"controller": {"P": 3}},
        )])
        outputs = iter([self.result(0.8), self.result(1.0)])
        with tempfile.TemporaryDirectory() as temp:
            report = self.run_loop(spec, planner, lambda current, path: next(outputs), temp)
            original_dir = Path(report.attempts[0].attempt_dir)
            saved_sim = (original_dir / "simulation.json").read_bytes()
            saved_verification = (original_dir / "verification.json").read_bytes()
            original_report = (Path(report.run_dir) / "repair_report.json").read_bytes()
            with self.assertRaises(FileExistsError):
                self.run_loop(spec, FakeRepairPlanner([]), lambda current, path: self.result(), temp)
            self.assertEqual((original_dir / "simulation.json").read_bytes(), saved_sim)
            self.assertEqual((original_dir / "verification.json").read_bytes(), saved_verification)
            self.assertEqual((Path(report.run_dir) / "repair_report.json").read_bytes(), original_report)

    def test_malformed_verification_is_terminal_error_and_never_triggers_repair(self):
        planner = FakeRepairPlanner([RepairProposal(
            reason="must not be reached", changes_summary="would change P", parameter_updates={"controller": {"P": 4}},
        )])
        with tempfile.TemporaryDirectory() as temp, patch("circuitpilot.repair.verify_simulation", return_value={"status": "pass"}):
            report = self.run_loop(self.make_spec(), planner, lambda current, path: self.result(0.8), temp)
            self.assertEqual(report.status, "error")
            self.assertEqual(planner.calls, 0)
            self.assertEqual(report.attempts[0].engineering_verification_status, "error")
            self.assertEqual(report.attempts[0].simulation_execution_status, "succeeded")
            saved = json.loads((Path(report.run_dir) / "repair_report.json").read_text())
            self.assertEqual(saved["status"], "error")
            self.assertTrue((Path(report.attempts[0].attempt_dir) / "failure.json").exists())

    def test_timeout_during_generation_is_reported_and_stops_without_repair(self):
        planner = FakeRepairPlanner([])
        with tempfile.TemporaryDirectory() as temp:
            report = self.run_loop(self.make_spec(), planner, lambda current, path: (_ for _ in ()).throw(TimeoutError()), temp)
            attempt = report.attempts[0]
            self.assertEqual(report.status, "error")
            self.assertEqual(attempt.failure_type, "timeout")
            self.assertEqual(attempt.simulation_execution_status, "timeout")
            self.assertEqual(attempt.engineering_verification_status, "fail")
            self.assertEqual(planner.calls, 0)

    def test_generator_result_must_be_structured_before_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            report = self.run_loop(self.make_spec(), FakeRepairPlanner([]), lambda current, path: ["succeeded"], temp)
            self.assertEqual(report.status, "error")
            self.assertEqual(report.attempts[0].simulation_execution_status, "unknown")
            self.assertEqual(report.attempts[0].engineering_verification_status, "error")
            self.assertTrue((Path(report.attempts[0].attempt_dir) / "failure.json").exists())

    def test_success_on_first_run_needs_no_repair(self):
        planner = FakeRepairPlanner([])
        calls = []

        def generator(spec, output_dir):
            calls.append(Path(output_dir))
            return self.result()

        with tempfile.TemporaryDirectory() as temp:
            report = self.run_loop(self.make_spec(), planner, generator, temp)
            self.assertEqual(report.status, "pass")
            self.assertEqual(len(calls), 1)
            self.assertEqual(planner.calls, 0)
            self.assertTrue((Path(report.run_dir) / "repair_report.json").exists())

    def test_failed_verification_then_targeted_repair_passes_and_preserves_original(self):
        original = self.make_spec()
        planner = FakeRepairPlanner([RepairProposal(
            reason="Increase proportional response to reduce the measured final error.",
            changes_summary="Increase controller proportional gain from 2 to 3.",
            parameter_updates={"controller": {"P": 3}},
        )])
        results = iter([self.result(0.8), self.result(1.0)])
        paths = []

        def generator(spec, output_dir):
            paths.append(Path(output_dir))
            return next(results)

        with tempfile.TemporaryDirectory() as temp:
            report = self.run_loop(original, planner, generator, temp)
            self.assertEqual(report.status, "pass")
            self.assertEqual([item.status for item in report.attempts], ["fail", "pass"])
            self.assertEqual(report.attempts[0].simulation_execution_status, "succeeded")
            self.assertEqual(report.attempts[0].engineering_verification_status, "fail")
            self.assertEqual(report.attempts[1].proposal_validation, "accepted")
            self.assertEqual(report.attempts[1].engineering_verification_status, "pass")
            self.assertEqual(planner.last_diagnosis.failed_checks[0].name, "final_value_tracking_error")
            self.assertAlmostEqual(planner.last_diagnosis.failed_checks[0].value, 0.2)
            self.assertEqual(original.components[2].parameters["P"], 2)
            self.assertEqual(planner.last_spec.components[2].parameters["P"], 2)
            self.assertNotEqual(paths[0], paths[1])
            self.assertTrue((paths[1] / "proposal.json").exists())
            stored_original = json.loads((paths[0] / "spec.json").read_text())
            stored_repair = json.loads((paths[1] / "spec.json").read_text())
            self.assertEqual(stored_original["components"][2]["parameters"]["P"], 2)
            self.assertEqual(stored_repair["components"][2]["parameters"]["P"], 3)
            markdown = (Path(report.run_dir) / "verification_report.md").read_text(encoding="utf-8")
            self.assertIn("Attempt 0", markdown)
            self.assertIn("Attempt 1", markdown)
            self.assertIn("<= 0.05 output units", markdown)

    def test_invalid_repair_is_rejected_before_generator_runs_again(self):
        planner = FakeRepairPlanner([RepairProposal(
            reason="Attempt to alter unsupported parameter.",
            changes_summary="Add an unsupported plant option.",
            parameter_updates={"plant": {"Unsupported": 1}},
        )])
        calls = []

        def generator(spec, output_dir):
            calls.append(spec)
            return self.result(0.8)

        with tempfile.TemporaryDirectory() as temp:
            report = self.run_loop(self.make_spec(), planner, generator, temp)
            self.assertEqual(report.status, "fail")
            self.assertEqual([attempt.status for attempt in report.attempts], ["fail", "rejected"])
            self.assertEqual(len(calls), 1)
            self.assertIn("validation", report.attempts[-1].failure_reason.lower())
            self.assertEqual(report.attempts[-1].proposal_validation, "rejected")
            self.assertEqual(report.attempts[-1].simulation_execution_status, "not_started")
            self.assertEqual(report.attempts[-1].engineering_verification_status, "not_run")
            self.assertTrue((Path(report.attempts[-1].attempt_dir) / "failure.json").exists())

    def test_attempt_limit_is_bounded_and_records_each_simulation(self):
        proposal = RepairProposal(
            reason="Try a bounded gain update.",
            changes_summary="Set P to 3.",
            parameter_updates={"controller": {"P": 3}},
        )
        planner = FakeRepairPlanner([proposal])
        calls = []

        def generator(spec, output_dir):
            calls.append(Path(output_dir))
            return self.result(0.8)

        with tempfile.TemporaryDirectory() as temp:
            report = self.run_loop(self.make_spec(), planner, generator, temp, max_attempts=1)
            self.assertEqual(report.status, "fail")
            self.assertEqual(len(calls), 2)
            self.assertEqual(planner.calls, 1)
            self.assertEqual([attempt.number for attempt in report.attempts], [0, 1])
            self.assertIn("limit reached", report.summary)

    def test_core_attempt_limit_accepts_zero_and_five_but_rejects_negative_excessive_and_missing_types(self):
        spec = self.make_spec()
        with tempfile.TemporaryDirectory() as temp:
            for value in (-1, 6, True, 1.5):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    run_repair_loop(spec, FakeRepairPlanner([]), lambda current, path: self.result(), temp,
                                    max_repair_attempts=value, run_id=f"bad-{str(value).replace('.', '_')}")
            zero = run_repair_loop(spec, FakeRepairPlanner([]), lambda current, path: self.result(), temp,
                                   max_repair_attempts=0, run_id="zero")
            self.assertEqual(zero.max_repair_attempts, 0)
            self.assertEqual(len(zero.attempts), 1)
            five = run_repair_loop(spec, FakeRepairPlanner([]), lambda current, path: self.result(), temp,
                                   max_repair_attempts=5, run_id="five")
            self.assertEqual(five.max_repair_attempts, 5)
            defaulted = run_repair_loop(self.make_spec(criteria=False), FakeRepairPlanner([]),
                                        lambda current, path: self.result(), temp, run_id="default")
            self.assertEqual(defaulted.max_repair_attempts, 2)

    def test_inconclusive_metrics_do_not_trigger_repair(self):
        planner = FakeRepairPlanner([])

        def generator(spec, output_dir):
            return self.result()

        with tempfile.TemporaryDirectory() as temp:
            report = self.run_loop(self.make_spec(criteria=False), planner, generator, temp)
            self.assertEqual(report.status, "inconclusive")
            self.assertEqual(planner.calls, 0)
            self.assertEqual(len(report.attempts), 1)
            self.assertIsNone(report.attempts[0].diagnosis)

    def test_generator_error_is_recorded_and_stops_safely(self):
        planner = FakeRepairPlanner([])

        def generator(spec, output_dir):
            raise RuntimeError("MATLAB failed to start")

        with tempfile.TemporaryDirectory() as temp:
            report = self.run_loop(self.make_spec(), planner, generator, temp)
            self.assertEqual(report.status, "error")
            self.assertEqual(report.attempts[0].status, "error")
            self.assertIn("MATLAB failed to start", report.attempts[0].failure_reason)
            self.assertEqual(planner.calls, 0)

    def test_transient_repair_planner_failure_is_recorded_without_second_generation(self):
        error = RuntimeError("temporary provider overload")
        error.status_code = 503
        error.response = type("Response", (), {
            "status_code": 503,
            "text": "high demand",
            "headers": {"Retry-After": "0"},
        })()

        class Interactions:
            calls = 0

            def create(self, **kwargs):
                self.calls += 1
                raise error

        client = type("Client", (), {"interactions": Interactions()})()
        planner = GeminiPlanner(client=client, max_retries=1)
        generation_calls = []

        def generator(spec, output_dir):
            generation_calls.append(Path(output_dir))
            return self.result(0.8)

        with tempfile.TemporaryDirectory() as temp, \
             patch("circuitpilot.planner.time.sleep"), \
             patch("circuitpilot.planner.random.uniform", return_value=0.1):
            report = self.run_loop(self.make_spec(), planner, generator, temp)
            self.assertEqual(report.status, "error")
            self.assertEqual(len(generation_calls), 1)
            self.assertEqual(client.interactions.calls, 2)
            self.assertEqual(len(report.attempts), 2)
            failed_check_names = {check.name for check in report.attempts[1].diagnosis.failed_checks}
            self.assertIn("final_value_tracking_error", failed_check_names)
            provider_attempt = report.attempts[1]
            self.assertEqual(provider_attempt.retry_count, 1)
            self.assertTrue(provider_attempt.transient_failure)
            self.assertIn("HTTP 503", provider_attempt.provider_error)
            saved_report = json.loads((Path(report.run_dir) / "repair_report.json").read_text())
            self.assertEqual(saved_report["attempts"][1]["retry_count"], 1)
            self.assertTrue(saved_report["attempts"][1]["transient_failure"])

    def test_planner_interruption_writes_final_report_and_preserves_diagnosis(self):
        class InterruptedPlanner:
            def propose_repair(self, spec, diagnosis):
                raise KeyboardInterrupt()

        generated = []

        def generator(spec, output_dir):
            generated.append(output_dir)
            return self.result(0.8)

        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(KeyboardInterrupt):
                self.run_loop(self.make_spec(), InterruptedPlanner(), generator, temp)
            self.assertEqual(len(generated), 1)
            run_dir = Path(temp) / "repairs" / "test-run"
            persisted = json.loads((run_dir / "repair_report.json").read_text())
            self.assertEqual(persisted["status"], "interrupted")
            self.assertEqual(persisted["attempts"][1]["failure_type"], "interrupted")
            self.assertEqual(
                persisted["attempts"][1]["diagnosis"]["failed_checks"][0]["name"],
                "final_value_tracking_error",
            )
            self.assertTrue((run_dir / "verification_report.md").exists())
            self.assertTrue((Path(persisted["attempts"][1]["attempt_dir"]) / "failure.json").exists())

    def test_unexpected_planner_exception_is_distinguished(self):
        planner = FakeRepairPlanner([RuntimeError("local adapter bug")])

        def generator(spec, output_dir):
            return self.result(0.8)

        with tempfile.TemporaryDirectory() as temp:
            report = self.run_loop(self.make_spec(), planner, generator, temp)
        failure = report.attempts[1]
        self.assertEqual(report.status, "error")
        self.assertEqual(failure.failure_type, "unexpected")
        self.assertIn("RuntimeError", failure.provider_error)
        self.assertNotIn("local adapter bug", failure.provider_error)
        self.assertEqual(failure.retry_count, 0)

    def test_gemini_timeout_is_saved_in_failure_and_final_reports(self):
        class TimeoutInteractions:
            calls = 0

            def create(self, **kwargs):
                self.calls += 1
                raise TimeoutError("SENSITIVE_TIMEOUT_DETAIL_MUST_NOT_BE_LOGGED")

        interactions = TimeoutInteractions()
        planner = GeminiPlanner(
            client=type("Client", (), {"interactions": interactions})(),
            max_retries=1,
        )
        generation_calls = []

        def generator(spec, output_dir):
            generation_calls.append(Path(output_dir))
            return self.result(0.8)

        with tempfile.TemporaryDirectory() as temp, \
             patch("circuitpilot.planner.time.sleep"), \
             patch("circuitpilot.planner.random.uniform", return_value=0.1), \
             self.assertLogs("circuitpilot.planner", level="INFO") as captured:
            report = self.run_loop(self.make_spec(), planner, generator, temp)
            self.assertEqual(report.status, "error")
            self.assertEqual(interactions.calls, 2)
            self.assertEqual(len(generation_calls), 1)
            self.assertEqual([attempt.status for attempt in report.attempts], ["fail", "error"])

            original = report.attempts[0]
            repair = report.attempts[1]
            self.assertIsNotNone(original.diagnosis)
            self.assertEqual(original.diagnosis.failed_checks, repair.diagnosis.failed_checks)
            self.assertEqual(repair.failure_type, "timeout")
            self.assertEqual(repair.retry_count, 1)
            self.assertTrue(repair.transient_failure)

            failure_path = Path(repair.attempt_dir) / "failure.json"
            failure = json.loads(failure_path.read_text())
            self.assertEqual(failure["failure_type"], "timeout")
            self.assertEqual(failure["retry_count"], 1)
            self.assertTrue(failure["transient_failure"])

            report_path = Path(report.run_dir) / "repair_report.json"
            saved = json.loads(report_path.read_text())
            self.assertEqual(saved["status"], "error")
            self.assertEqual(
                saved["attempts"][0]["diagnosis"]["failed_checks"],
                saved["attempts"][1]["diagnosis"]["failed_checks"],
            )
            self.assertEqual(saved["attempts"][1]["retry_count"], 1)
            self.assertTrue(saved["attempts"][1]["transient_failure"])
            self.assertTrue(any("Gemini repair request failed" in line for line in captured.output))
            self.assertNotIn("SENSITIVE_TIMEOUT_DETAIL_MUST_NOT_BE_LOGGED", "\n".join(captured.output))

    def test_checkpoint_survives_an_unhandled_process_level_abort(self):
        class SimulatedHardStop(BaseException):
            pass

        class StoppedPlanner:
            def propose_repair(self, spec, diagnosis):
                raise SimulatedHardStop("simulated process-level abort")

        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(SimulatedHardStop):
                self.run_loop(
                    self.make_spec(), StoppedPlanner(),
                    lambda spec, output_dir: self.result(0.8), temp,
                )
            checkpoint = json.loads(
                (Path(temp) / "repairs" / "test-run" / "repair_report.json").read_text()
            )
            self.assertEqual(checkpoint["status"], "running")
            self.assertEqual(checkpoint["attempts"][-1]["status"], "planning")
            self.assertEqual(
                checkpoint["attempts"][-1]["diagnosis"]["failed_checks"][0]["name"],
                "final_value_tracking_error",
            )

    def test_cli_enables_repair_loop_and_accepts_attempt_limit(self):
        spec = self.make_spec(criteria=False)
        planner = FakeRepairPlanner([])
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp:
            spec_path = Path(temp) / "spec.json"
            spec_path.write_text(spec.model_dump_json(), encoding="utf-8")
            with patch("sys.argv", [
                "circuitpilot", str(spec_path), "--output", temp,
                "--auto-repair", "--max-repair-attempts", "1",
            ]), patch.dict(os.environ, {"CIRCUITPILOT_PROVIDER": "openai"}), \
                 patch("circuitpilot.cli.OpenAIPlanner", return_value=planner), \
                 patch("circuitpilot.cli.generate_and_simulate", return_value=self.result()), \
                 redirect_stdout(output):
                main()
            rendered = json.loads(output.getvalue())
            self.assertEqual(rendered["status"], "inconclusive")
            self.assertEqual(rendered["max_repair_attempts"], 1)
            self.assertEqual(planner.calls, 0)


if __name__ == "__main__":
    unittest.main()
