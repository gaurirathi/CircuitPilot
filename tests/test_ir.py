import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from circuitpilot.ir import Component, ModelSpec, validate_model


class ComponentTests(unittest.TestCase):
    def test_registry_covers_requested_component_types(self):
        examples = {
            "Step": {"Time": 0, "Before": 0, "After": 1},
            "Sum": {"Inputs": "+-"},
            "Gain": {"Gain": 2},
            "Transfer Fcn": {"Numerator": [1], "Denominator": [1, 1]},
            "PID Controller": {"P": 1, "I": 0, "D": 0},
            "Integrator": {"InitialCondition": 0},
            "State-Space": {"A": [[-1]], "B": [[1]], "C": [[1]], "D": [[0]]},
            "Scope": {},
        }
        for kind, parameters in examples.items():
            with self.subTest(kind=kind):
                Component(id="block", type=kind, parameters=parameters)

    def test_unknown_type_and_parameter_are_rejected(self):
        with self.assertRaisesRegex(
            ValidationError,
            r"Unsupported component type 'MOSFET'.*Supported types: Step, Sum, Gain, Transfer Fcn, PID Controller, Integrator, State-Space, Scope",
        ):
            Component(id="x", type="MOSFET")
        with self.assertRaises(ValidationError):
            Component(id="x", type="Gain", parameters={"Bogus": 2})

    def test_invalid_transfer_function_is_rejected(self):
        with self.assertRaises(ValidationError):
            Component(id="plant", type="Transfer Fcn", parameters={"Denominator": [0, 0]})

    def test_required_component_parameters_are_rejected_when_missing(self):
        with self.assertRaisesRegex(ValidationError, "Missing required Transfer Fcn parameter"):
            Component(id="plant", type="Transfer Fcn", parameters={"Numerator": [1]})


class GraphTests(unittest.TestCase):
    def setUp(self):
        data = json.loads((Path(__file__).parents[1] / "examples/closed_loop_pid.json").read_text())
        self.spec = ModelSpec.model_validate(data)

    def test_closed_loop_pid_graph_is_valid(self):
        validate_model(self.spec)

    def test_duplicate_input_driver_is_rejected(self):
        data = self.spec.model_dump()
        data["connections"].append({"source": "reference", "target": "controller"})
        with self.assertRaisesRegex(ValueError, "multiple drivers"):
            validate_model(ModelSpec.model_validate(data))

    def test_bad_port_and_unknown_node_are_rejected(self):
        data = self.spec.model_dump()
        data["connections"][0]["target_port"] = 3
        with self.assertRaisesRegex(ValueError, "Target port out of range"):
            validate_model(ModelSpec.model_validate(data))
        data = self.spec.model_dump()
        data["connections"][0]["target"] = "missing"
        with self.assertRaisesRegex(ValueError, "unknown component"):
            validate_model(ModelSpec.model_validate(data))

    def test_model_settings_and_output_are_validated(self):
        with self.assertRaises(ValidationError):
            ModelSpec.model_validate({**self.spec.model_dump(), "stop_time": 0})
        with self.assertRaises(ValidationError):
            ModelSpec.model_validate({**self.spec.model_dump(), "stop_time": float("inf")})
        with self.assertRaisesRegex(ValueError, "does not exist"):
            validate_model(ModelSpec.model_validate({**self.spec.model_dump(), "output_component": "missing"}))

    def test_state_space_dimensions_are_validated(self):
        with self.assertRaises(ValidationError):
            Component(id="plant", type="State-Space", parameters={
                "A": [[-1, 0], [0, -2]], "B": [[1]], "C": [[1]], "D": [[0]]
            })

    def test_matlab_builder_uses_simulink_paths_for_destinations_and_sources(self):
        script = (Path(__file__).parents[1] / "circuitpilot/matlab/circuitpilot_generate_and_simulate.m").read_text()
        self.assertIn("blockPath = [model '/' char(component.id)]", script)
        self.assertNotIn("blockPath = fullfile(model", script)
        for library_path in (
            "simulink/Sources/Step",
            "simulink/Math Operations/Sum",
            "simulink/Continuous/PID Controller",
            "simulink/Continuous/Transfer Fcn",
            "simulink/Sinks/Scope",
        ):
            with self.subTest(library_path=library_path):
                self.assertIn(library_path, script)


if __name__ == "__main__":
    unittest.main()
