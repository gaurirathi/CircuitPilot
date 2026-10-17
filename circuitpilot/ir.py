"""Typed intermediate representation and validation for control models."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Component(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    type: str
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def supported_type(cls, value: str) -> str:
        if value not in COMPONENTS:
            supported = ", ".join(COMPONENTS)
            raise ValueError(
                f"Unsupported component type '{value}'. Supported types: {supported}."
            )
        return value

    @model_validator(mode="after")
    def validate_parameters(self) -> "Component":
        validate_component_parameters(self.type, self.parameters)
        return self


class Connection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    source_port: int = Field(default=1, ge=1)
    target: str
    target_port: int = Field(default=1, ge=1)


class VerificationCriteria(BaseModel):
    """Optional user-defined performance limits for a simulated response."""

    model_config = ConfigDict(extra="forbid")

    expected_final_value: float | None = None
    max_tracking_error: float | None = Field(default=None, ge=0)
    max_overshoot_percent: float | None = Field(default=None, ge=0)
    max_settling_time: float | None = Field(default=None, ge=0)
    settling_band_percent: float = Field(default=2.0, gt=0, lt=100)

    @field_validator("expected_final_value", "max_tracking_error", "max_overshoot_percent", "max_settling_time")
    @classmethod
    def finite_criteria(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("verification criteria must be finite")
        return value


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="CircuitPilotModel", pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    components: list[Component] = Field(min_length=1)
    connections: list[Connection] = Field(default_factory=list)
    stop_time: float = Field(default=10.0, gt=0)
    output_component: str
    output_port: int = Field(default=1, ge=1)
    verification: VerificationCriteria | None = None

    @field_validator("stop_time")
    @classmethod
    def finite_stop_time(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("stop_time must be finite")
        return value


@dataclass(frozen=True)
class _ComponentDefinition:
    library: str
    inputs: int | str
    outputs: int | str
    allowed_parameters: frozenset[str]


COMPONENTS: dict[str, _ComponentDefinition] = {
    "Step": _ComponentDefinition("simulink/Sources/Step", 0, 1, frozenset({"Time", "Before", "After"})),
    "Sum": _ComponentDefinition("simulink/Math Operations/Sum", "sum", 1, frozenset({"Inputs"})),
    "Gain": _ComponentDefinition("simulink/Math Operations/Gain", 1, 1, frozenset({"Gain"})),
    "Transfer Fcn": _ComponentDefinition("simulink/Continuous/Transfer Fcn", 1, 1, frozenset({"Numerator", "Denominator"})),
    "PID Controller": _ComponentDefinition("simulink/Continuous/PID Controller", 1, 1, frozenset({"P", "I", "D", "N", "Controller"})),
    "Integrator": _ComponentDefinition("simulink/Continuous/Integrator", 1, 1, frozenset({"InitialCondition"})),
    "State-Space": _ComponentDefinition("simulink/Continuous/State-Space", "state", "state", frozenset({"A", "B", "C", "D", "X0"})),
    "Scope": _ComponentDefinition("simulink/Sinks/Scope", 1, 0, frozenset()),
}

REQUIRED_PARAMETERS: dict[str, frozenset[str]] = {
    "Step": frozenset({"Time", "Before", "After"}),
    "Sum": frozenset({"Inputs"}),
    "Gain": frozenset({"Gain"}),
    "Transfer Fcn": frozenset({"Numerator", "Denominator"}),
    "PID Controller": frozenset({"P", "I", "D"}),
    "Integrator": frozenset({"InitialCondition"}),
    "State-Space": frozenset({"A", "B", "C", "D"}),
    "Scope": frozenset(),
}


def _finite_number(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")


def validate_component_parameters(kind: str, params: dict[str, Any]) -> None:
    definition = COMPONENTS[kind]
    unknown = set(params) - definition.allowed_parameters
    if unknown:
        raise ValueError(f"Unsupported {kind} parameter(s): {', '.join(sorted(unknown))}")
    missing = REQUIRED_PARAMETERS[kind] - set(params)
    if missing:
        raise ValueError(f"Missing required {kind} parameter(s): {', '.join(sorted(missing))}")
    scalar_keys = ("Time", "Before", "After", "Gain", "P", "I", "N", "InitialCondition")
    if kind == "PID Controller":
        scalar_keys += ("D",)
    for key in scalar_keys:
        if key in params:
            _finite_number(params[key], f"{kind}.{key}")
    if "Time" in params and params["Time"] < 0:
        raise ValueError("Step.Time must be nonnegative")
    if "Controller" in params and params["Controller"] not in {"PID", "PI", "PD", "P"}:
        raise ValueError("PID Controller.Controller must be PID, PI, PD, or P")
    if "Inputs" in params:
        signs = params["Inputs"]
        if not isinstance(signs, str) or not signs or re.fullmatch(r"[+-]+", signs) is None:
            raise ValueError("Sum.Inputs must be a nonempty string of + and - signs")
    for key in ("Numerator", "Denominator"):
        if key in params:
            values = params[key]
            if not isinstance(values, list) or not values:
                raise ValueError(f"{kind}.{key} must be a nonempty coefficient list")
            for i, value in enumerate(values):
                _finite_number(value, f"{kind}.{key}[{i}]")
    if "Denominator" in params and all(float(v) == 0 for v in params["Denominator"]):
        raise ValueError("Transfer Fcn.Denominator cannot be all zero")
    if kind == "State-Space":
        if any(key in params for key in ("A", "B", "C", "D")) and not all(
            key in params for key in ("A", "B", "C", "D")
        ):
            raise ValueError("State-Space requires A, B, C, and D together")
        for key in ("A", "B", "C", "D"):
            if key in params and (not isinstance(params[key], list) or not params[key] or any(not isinstance(row, list) or not row for row in params[key])):
                raise ValueError(f"State-Space.{key} must be a nonempty 2D list")
            if key in params:
                for row_i, row in enumerate(params[key]):
                    for col_i, value in enumerate(row):
                        _finite_number(value, f"State-Space.{key}[{row_i}][{col_i}]")
        if all(key in params for key in ("A", "B", "C", "D")):
            a, b, c, d = (params[k] for k in ("A", "B", "C", "D"))
            n = len(a)
            if len(a) != n or any(len(row) != n for row in a):
                raise ValueError("State-Space.A must be square")
            if len(b) != n or len(c[0]) != n:
                raise ValueError("State-Space A, B, and C dimensions are incompatible")
            if any(len(row) != len(b[0]) for row in b) or any(len(row) != n for row in c):
                raise ValueError("State-Space B or C dimensions are inconsistent")
            if len(d) != len(c) or any(len(row) != len(b[0]) for row in d):
                raise ValueError("State-Space.D dimensions are incompatible with B and C")
            if "X0" in params:
                x0 = params["X0"]
                if not isinstance(x0, list) or len(x0) != n:
                    raise ValueError(f"State-Space.X0 must contain {n} state values")
                for index, value in enumerate(x0):
                    _finite_number(value, f"State-Space.X0[{index}]")


def _port_counts(component: Component) -> tuple[int, int]:
    definition = COMPONENTS[component.type]
    if component.type == "Sum":
        return len(component.parameters.get("Inputs", "+-")), 1
    if component.type == "State-Space" and all(k in component.parameters for k in ("A", "B", "C", "D")):
        return len(component.parameters["B"][0]), len(component.parameters["C"])
    return int(definition.inputs), int(definition.outputs)


def validate_model(spec: ModelSpec) -> None:
    """Raise ValueError if graph references or port cardinalities are invalid."""
    by_id = {component.id: component for component in spec.components}
    if len(by_id) != len(spec.components):
        raise ValueError("Component ids must be unique")
    if spec.output_component not in by_id:
        raise ValueError(f"Output component '{spec.output_component}' does not exist")
    output_component = by_id[spec.output_component]
    if spec.output_port > _port_counts(output_component)[1]:
        raise ValueError("Output port is out of range")
    occupied_inputs: set[tuple[str, int]] = set()
    for connection in spec.connections:
        if connection.source not in by_id or connection.target not in by_id:
            raise ValueError("Connection references an unknown component")
        source, target = by_id[connection.source], by_id[connection.target]
        if connection.source_port > _port_counts(source)[1]:
            raise ValueError(f"Source port out of range on '{source.id}'")
        if connection.target_port > _port_counts(target)[0]:
            raise ValueError(f"Target port out of range on '{target.id}'")
        port = (connection.target, connection.target_port)
        if port in occupied_inputs:
            raise ValueError(f"Input port {connection.target_port} on '{connection.target}' has multiple drivers")
        occupied_inputs.add(port)
    for component in spec.components:
        count, _ = _port_counts(component)
        for port in range(1, count + 1):
            if (component.id, port) not in occupied_inputs:
                raise ValueError(f"Input port {port} on '{component.id}' is not connected")
    if (spec.output_component, spec.output_port) not in {
        (c.source, c.source_port) for c in spec.connections
    }:
        raise ValueError("Selected output port has no incoming signal")

