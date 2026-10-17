"""MATLAB Engine bridge. Importing this module does not require MATLAB Engine."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from .ir import ModelSpec, validate_model


def generate_and_simulate(spec: ModelSpec, output_dir: str | Path) -> dict[str, Any]:
    """Generate an SLX model and run it through a locally installed MATLAB Engine."""
    validate_model(spec)
    try:
        import matlab.engine  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "MATLAB Engine for Python is not installed. Install it from the active MATLAB "
            "release's extern/engines/python directory."
        ) from exc

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="circuitpilot-") as temp_dir:
        temp = Path(temp_dir)
        spec_file, result_file = temp / "spec.json", temp / "result.json"
        spec_file.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        matlab_root = Path(__file__).parent / "matlab"
        engine = None
        phase = "MATLAB Engine startup"
        try:
            engine = matlab.engine.start_matlab()
            phase = "MATLAB path setup"
            engine.addpath(str(matlab_root), nargout=0)
            phase = "CircuitPilot MATLAB function (model generation and simulation)"
            result_json = engine.circuitpilot_generate_and_simulate(
                str(spec_file), str(output_path), str(result_file), nargout=1
            )
            phase = "simulation result retrieval"
            if result_file.exists():
                return json.loads(result_file.read_text(encoding="utf-8"))
            return json.loads(str(result_json))
        except Exception as exc:
            raise RuntimeError(f"{phase} failed ({type(exc).__name__}): {exc}") from exc
        finally:
            if engine is not None:
                active_error = sys.exc_info()[1]
                try:
                    engine.quit()
                except Exception as quit_error:
                    if active_error is None:
                        raise RuntimeError(
                            f"MATLAB Engine shutdown failed ({type(quit_error).__name__}): {quit_error}"
                        ) from quit_error
                    # A dead Engine transport can make quit fail too. Preserve
                    # the operation failure and retain shutdown detail on it.
                    try:
                        active_error.engine_shutdown_error = (
                            f"{type(quit_error).__name__}: {quit_error}"
                        )
                    except Exception:
                        pass

