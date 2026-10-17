"""Standalone staged diagnostics for the local MATLAB Engine installation.

Run with ``python -m circuitpilot.matlab_diagnostic`` for an Engine startup,
expression, and shutdown check. Add ``--generator`` to run CircuitPilot's MATLAB
function with a tiny Step-to-Gain model after the Engine checks pass.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from .ir import ModelSpec, validate_model


def _minimal_spec() -> ModelSpec:
    """Use the same validated IR and default port serialization as production."""
    spec = ModelSpec.model_validate({
        "name": "CircuitPilotEngineDiagnostic",
        "components": [
            {"id": "step", "type": "Step", "parameters": {"Time": 0, "Before": 0, "After": 1}},
            {"id": "gain", "type": "Gain", "parameters": {"Gain": 2}},
            {"id": "scope", "type": "Scope", "parameters": {}},
        ],
        "connections": [
            {"source": "step", "target": "gain"},
            {"source": "gain", "target": "scope"},
        ],
        "stop_time": 1,
        "output_component": "gain",
    })
    validate_model(spec)
    return spec


def run_diagnostic(engine_module: Any, *, run_generator: bool = False) -> dict[str, Any]:
    """Exercise Engine startup/eval/quit, then optionally the model generator."""
    report: dict[str, Any] = {
        "startup": "not_run",
        "expression": "not_run",
        "generator_function": "not_requested" if not run_generator else "not_run",
        "shutdown": "not_run",
    }
    engine = None
    try:
        report["phase"] = "startup"
        engine = engine_module.start_matlab()
        report["startup"] = "succeeded"

        report["phase"] = "expression"
        report["matlab_version"] = str(engine.eval("version", nargout=1))
        report["expression"] = "succeeded"

        if run_generator:
            report["phase"] = "generator_function"
            package_root = Path(__file__).resolve().parent
            matlab_dir = package_root / "matlab"
            with tempfile.TemporaryDirectory(prefix="circuitpilot-matlab-diagnostic-") as temp_dir:
                temp = Path(temp_dir)
                output_dir = temp / "output"
                output_dir.mkdir()
                spec_file = temp / "spec.json"
                result_file = temp / "result.json"
                spec_file.write_text(_minimal_spec().model_dump_json(indent=2), encoding="utf-8")
                engine.addpath(str(matlab_dir), nargout=0)
                result_json = engine.circuitpilot_generate_and_simulate(
                    str(spec_file), str(output_dir), str(result_file), nargout=1
                )
                result = (
                    json.loads(result_file.read_text(encoding="utf-8"))
                    if result_file.exists()
                    else json.loads(str(result_json))
                )
                report["model_saved"] = Path(result["model_path"]).is_file()
                report["simulation_status"] = result.get("simulation_status")
                report["sample_count"] = len(result.get("time", []))
                if not report["model_saved"] or report["simulation_status"] != "succeeded":
                    raise RuntimeError("CircuitPilot MATLAB function returned without a saved, successful simulation.")
                report["generator_function"] = "succeeded"
    except Exception as exc:
        failed_phase = report.get("phase", "unknown")
        report["failed_phase"] = failed_phase
        if failed_phase in {"startup", "expression"}:
            report[failed_phase] = "failed"
        elif failed_phase == "generator_function":
            report["generator_function"] = "failed"
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
    finally:
        if engine is not None:
            try:
                engine.quit()
                report["shutdown"] = "succeeded"
            except Exception as exc:
                report["shutdown"] = "failed"
                report["shutdown_error_type"] = type(exc).__name__
                report["shutdown_error"] = str(exc)
                if report.get("status") != "failed":
                    report["status"] = "failed"
                    report["failed_phase"] = "shutdown"

    if "status" not in report:
        report["status"] = "succeeded"
    if report["status"] == "succeeded":
        report["phase"] = "complete"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose MATLAB Engine startup and CircuitPilot MATLAB execution.")
    parser.add_argument(
        "--generator", action="store_true",
        help="After Engine startup and eval pass, run the MATLAB generator with a minimal Step-to-Gain model",
    )
    args = parser.parse_args()
    try:
        import matlab.engine  # type: ignore[import-not-found]
    except ImportError as exc:
        print(json.dumps({"status": "failed", "phase": "python_import", "error_type": type(exc).__name__, "error": str(exc)}, indent=2))
        return 1

    report = run_diagnostic(matlab.engine, run_generator=args.generator)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
