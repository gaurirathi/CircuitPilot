"""Deterministic, MATLAB-backed Simulink model generation."""

from .ir import Component, Connection, ModelSpec, validate_model

__all__ = ["Component", "Connection", "ModelSpec", "validate_model"]
