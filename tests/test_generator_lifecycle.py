import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from circuitpilot.generator import generate_and_simulate
from circuitpilot.ir import ModelSpec


class GeneratorLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec_path = Path(__file__).parents[1] / "examples" / "closed_loop_pid.json"
        cls.spec = ModelSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))

    @staticmethod
    def fake_matlab_module(engine_module):
        matlab_module = types.ModuleType("matlab")
        matlab_module.engine = engine_module
        return {"matlab": matlab_module, "matlab.engine": engine_module}

    def test_startup_transport_error_is_labeled_and_not_misreported_as_model_failure(self):
        engine_module = types.ModuleType("matlab.engine")

        def fail_start():
            raise RuntimeError("Transport stopped")

        engine_module.start_matlab = fail_start
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, self.fake_matlab_module(engine_module)):
            with self.assertRaisesRegex(RuntimeError, "MATLAB Engine startup failed.*Transport stopped"):
                generate_and_simulate(self.spec, temp)

    def test_python_generator_serializes_connection_ports_for_matlab(self):
        class CapturingEngine:
            def addpath(self, path, nargout):
                pass

            def circuitpilot_generate_and_simulate(self, spec_path, output_dir, result_path, nargout):
                self.spec_payload = json.loads(Path(spec_path).read_text(encoding="utf-8"))
                Path(result_path).write_text(json.dumps({"simulation_status": "diagnostic"}))
                return "{}"

            def quit(self):
                pass

        engine = CapturingEngine()
        engine_module = types.ModuleType("matlab.engine")
        engine_module.start_matlab = lambda: engine
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, self.fake_matlab_module(engine_module)):
            generate_and_simulate(self.spec, temp)
        self.assertTrue(engine.spec_payload["connections"])
        for connection in engine.spec_payload["connections"]:
            self.assertIn("source_port", connection)
            self.assertIn("target_port", connection)
            self.assertGreaterEqual(connection["source_port"], 1)
            self.assertGreaterEqual(connection["target_port"], 1)

    def test_function_error_is_preserved_if_engine_quit_also_fails(self):
        class FakeEngine:
            def addpath(self, path, nargout):
                pass

            def circuitpilot_generate_and_simulate(self, *args, **kwargs):
                raise RuntimeError("original MATLAB transport failure")

            def quit(self):
                raise RuntimeError("secondary quit transport failure")

        engine_module = types.ModuleType("matlab.engine")
        engine_module.start_matlab = lambda: FakeEngine()
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, self.fake_matlab_module(engine_module)):
            with self.assertRaisesRegex(RuntimeError, "CircuitPilot MATLAB function.*original MATLAB transport failure") as raised:
                generate_and_simulate(self.spec, temp)
        self.assertIn("secondary quit transport failure", raised.exception.engine_shutdown_error)


if __name__ == "__main__":
    unittest.main()
