"""Reusable baseline planner-to-simulation workflow."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from .generator import generate_and_simulate
from .ir import ModelSpec, validate_model
from .planner import GeminiPlanner, OpenAIPlanner, PlannerError
from .run_reports import RunReportWriter
from .verification import verify_simulation


def make_planner(provider: str | None = None, model: str | None = None) -> Any:
    selected = (provider or os.getenv("CIRCUITPILOT_PROVIDER", "openai")).strip().lower()
    if selected == "gemini":
        return GeminiPlanner(model=model)
    return OpenAIPlanner(provider=selected, model=model)


def run_baseline(
    *,
    output_dir: str | Path,
    writer: RunReportWriter,
    spec: ModelSpec | None = None,
    prompt: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    duration: float | None = None,
    graph_output: str | Path | None = None,
    on_spec: Callable[[ModelSpec], None] | None = None,
    planner: Any | None = None,
    generator: Callable[[ModelSpec, str | Path], dict[str, Any]] = generate_and_simulate,
    on_stage: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run the non-repair workflow and update the supplied persistent report."""
    def stage(name: str) -> None:
        check_deadline()
        writer.update(stage=name, lifecycle_status="running")
        if on_stage:
            on_stage(name)

    def check_deadline() -> None:
        if should_stop is not None and should_stop():
            raise TimeoutError("CircuitPilot API execution deadline exceeded")

    try:
        if (spec is None) == (prompt is None):
            raise ValueError("provide exactly one of spec or prompt")
        if prompt is not None:
            stage("planning")
            planner = planner or make_planner(provider, model)
            spec = planner.plan(prompt)
            check_deadline()
        assert spec is not None
        if duration is not None:
            spec = ModelSpec.model_validate({**spec.model_dump(mode="python"), "stop_time": duration})
        stage("validation")
        validate_model(spec)
        if on_spec:
            on_spec(spec)
        if graph_output is not None or prompt is not None:
            graph_path = Path(graph_output) if graph_output is not None else Path(output_dir) / f"{spec.name}.graph.json"
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            graph_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        stage("generation")
        writer.update(model_generation_status="in progress", matlab_startup_status="in progress", simulation_status="in progress")
        stage("simulation")
        result = generator(spec, output_dir)
        check_deadline()
        stage("verification")
        verification = verify_simulation(spec, result)
        check_deadline()
        result["verification"] = verification.model_dump(mode="json")
        writer.set_result(result, result["verification"])
        if on_stage:
            on_stage("completed")
        return result
    except KeyboardInterrupt as exc:
        if should_stop is not None and should_stop():
            raise TimeoutError("CircuitPilot API execution deadline exceeded") from None
        writer.set_error(exc, stage=writer.data.get("stage", "unknown"), interrupted=True)
        if on_stage:
            on_stage("interrupted")
        raise
    except Exception as exc:
        if should_stop is not None and should_stop():
            # The API watchdog already wrote the terminal timeout checkpoint.
            raise TimeoutError("CircuitPilot API execution deadline exceeded") from None
        failed_stage = writer.data.get("stage", "unknown")
        if failed_stage in {"generation", "simulation"}:
            failed_stage = "model generation and simulation"
        writer.set_error(exc, stage=failed_stage)
        if on_stage:
            on_stage("failed")
        raise


def run_repair_baseline(
    *, output_dir: str | Path, writer: RunReportWriter, prompt: str, planner: Any,
    duration: float, generator: Callable[[ModelSpec, str | Path], dict[str, Any]],
    run_id: str, max_repair_attempts: int, on_stage: Callable[[str], None] | None = None,
    on_spec: Callable[[ModelSpec], None] | None = None, should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Plan once, then reuse the validated bounded repair loop for isolated attempts."""
    from .repair import run_repair_loop
    import json

    def stage(name: str) -> None:
        if should_stop is not None and should_stop():
            raise TimeoutError("CircuitPilot API execution deadline exceeded")
        writer.update(stage=name, lifecycle_status="running")
        if on_stage:
            on_stage(name)

    stage("planning")
    spec = planner.plan(prompt)
    if duration is not None:
        spec = ModelSpec.model_validate({**spec.model_dump(mode="python"), "stop_time": duration})
    stage("validation")
    validate_model(spec)
    if on_spec:
        on_spec(spec)
    (Path(output_dir) / f"{spec.name}.graph.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    writer.update(repair={"enabled": True, "attempt_count": 0,
                          "max_attempts": max_repair_attempts, "outcome": "running"})

    loop = run_repair_loop(
        spec, planner, generator, output_dir, max_repair_attempts=max_repair_attempts,
        run_id=run_id, input_path=writer.data.get("input_path"), on_stage=stage,
        should_stop=should_stop,
    )
    repair_payload = loop.model_dump(mode="json")
    attempts = repair_payload["attempts"]
    selected = next((attempt for attempt in attempts if attempt["number"] == loop.best_attempt), None)
    if selected is None:
        selected = next((attempt for attempt in reversed(attempts)
                         if attempt.get("specification_path")
                         and (Path(attempt["attempt_dir"]) / "simulation.json").is_file()
                         and (Path(attempt["attempt_dir"]) / "verification.json").is_file()), None)
    if selected is None:
        writer.update(repair={
            "enabled": True, "attempt_count": sum(attempt["kind"] == "repair" for attempt in attempts),
            "max_attempts": max_repair_attempts, "outcome": loop.status,
            "best_attempt": loop.best_attempt, "summary": loop.summary, "attempts": attempts,
        })
        raise RuntimeError("Repair workflow did not produce a usable simulation attempt")
    attempt_dir = Path(selected["attempt_dir"])
    result = json.loads((attempt_dir / "simulation.json").read_text(encoding="utf-8"))
    verification = json.loads((attempt_dir / "verification.json").read_text(encoding="utf-8"))
    best_spec = ModelSpec.model_validate_json(Path(selected["specification_path"]).read_text(encoding="utf-8"))
    if on_spec:
        on_spec(best_spec)
    result["verification"] = verification
    writer.set_result(result, verification)
    writer.update(repair={
        "enabled": True,
        "attempt_count": sum(attempt["kind"] == "repair" for attempt in attempts),
        "max_attempts": max_repair_attempts,
        "outcome": loop.status,
        "best_attempt": loop.best_attempt,
        "summary": loop.summary,
        "attempts": attempts,
    })
    if loop.status in {"error", "interrupted"}:
        if loop.status == "interrupted":
            raise KeyboardInterrupt
        raise RuntimeError("Repair planning or execution failed; see the safe repair report for details")
    if on_stage:
        on_stage("completed")
    return result
