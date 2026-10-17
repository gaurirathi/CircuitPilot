import json
import unittest
from pathlib import Path

from circuitpilot.matlab_diagnostic import _minimal_spec, run_diagnostic


class EngineDiagnosticTests(unittest.TestCase):
    def test_minimal_diagnostic_uses_validated_connection_port_defaults(self):
        connections = json.loads(_minimal_spec().model_dump_json())["connections"]
        self.assertEqual(connections, [
            {"source": "step", "source_port": 1, "target": "gain", "target_port": 1},
            {"source": "gain", "source_port": 1, "target": "scope", "target_port": 1},
        ])

    def test_simple_expression_and_quit_are_staged(self):
        class FakeEngine:
            def eval(self, expression, nargout):
                self.expression = expression
                self.nargout = nargout
                return "R2024b"

            def quit(self):
                self.quitted = True

        engine = FakeEngine()
        report = run_diagnostic(type("EngineModule", (), {"start_matlab": lambda: engine}))
        self.assertEqual(report["status"], "succeeded")
        self.assertEqual(report["startup"], "succeeded")
        self.assertEqual(report["expression"], "succeeded")
        self.assertEqual(report["matlab_version"], "R2024b")
        self.assertEqual(report["shutdown"], "succeeded")
        self.assertTrue(engine.quitted)

    def test_startup_failure_is_reported_at_startup(self):
        def fail_start():
            raise RuntimeError("Transport stopped")

        report = run_diagnostic(type("EngineModule", (), {"start_matlab": staticmethod(fail_start)}))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["phase"], "startup")
        self.assertEqual(report["error"], "Transport stopped")
        self.assertEqual(report["startup"], "failed")
        self.assertEqual(report["expression"], "not_run")

    def test_generator_failure_keeps_its_phase_and_still_quits_engine(self):
        class FakeEngine:
            def eval(self, expression, nargout):
                return "R2024b"

            def addpath(self, path, nargout):
                pass

            def circuitpilot_generate_and_simulate(self, *args, **kwargs):
                raise RuntimeError("Simulink diagnostic failure")

            def quit(self):
                self.quitted = True

        engine = FakeEngine()
        report = run_diagnostic(type("EngineModule", (), {"start_matlab": lambda: engine}), run_generator=True)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failed_phase"], "generator_function")
        self.assertEqual(report["generator_function"], "failed")
        self.assertTrue(engine.quitted)

    def test_minimal_generator_call_is_separate_and_checks_saved_simulation(self):
        class FakeEngine:
            def eval(self, expression, nargout):
                return "R2024b"

            def addpath(self, path, nargout):
                self.matlab_path = path

            def circuitpilot_generate_and_simulate(self, spec_path, output_dir, result_path, nargout):
                spec = json.loads(Path(spec_path).read_text())
                self.assertion = spec["components"][1]["type"] == "Gain"
                model_path = Path(output_dir) / "CircuitPilotEngineDiagnostic.slx"
                model_path.write_bytes(b"diagnostic placeholder")
                Path(result_path).write_text(json.dumps({
                    "model_path": str(model_path), "simulation_status": "succeeded",
                    "time": [0, 1], "output": [0, 2],
                }))
                return "{}"

            def quit(self):
                self.quitted = True

        engine = FakeEngine()
        report = run_diagnostic(type("EngineModule", (), {"start_matlab": lambda: engine}), run_generator=True)
        self.assertEqual(report["generator_function"], "succeeded")
        self.assertEqual(report["sample_count"], 2)
        self.assertTrue(engine.assertion)
        self.assertTrue(engine.quitted)


if __name__ == "__main__":
    unittest.main()
