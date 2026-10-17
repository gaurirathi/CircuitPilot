"""Persistent, sanitized human-readable reports for CircuitPilot runs."""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_report_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _status(value: str | None) -> str:
    return {
        "pass": "PASS",
        "fail": "FAIL",
        "inconclusive": "INCONCLUSIVE",
        "error": "ERROR",
        "interrupted": "ERROR (INTERRUPTED)",
        "running": "INCONCLUSIVE (IN PROGRESS)",
        "planning": "INCONCLUSIVE (IN PROGRESS)",
        "simulating": "INCONCLUSIVE (IN PROGRESS)",
        "rejected": "FAIL",
    }.get(value or "", "INCONCLUSIVE")


def _overall_status(value: str | None) -> str:
    """Keep the report's overall status within the documented four-state set."""
    return {
        "pass": "PASS",
        "fail": "FAIL",
        "inconclusive": "INCONCLUSIVE",
        "error": "ERROR",
        "interrupted": "ERROR",
        "running": "INCONCLUSIVE",
        "planning": "INCONCLUSIVE",
        "simulating": "INCONCLUSIVE",
        "rejected": "FAIL",
    }.get(value or "", "INCONCLUSIVE")


def _safe_phase(exc: BaseException) -> tuple[str | None, str | None]:
    """Extract only the known operation phase and exception class, never its message."""
    message = str(exc)
    match = re.match(r"^(.*?) failed \(([A-Za-z][A-Za-z0-9_.]*)\):", message)
    if match:
        known_phases = {
            "MATLAB Engine startup",
            "MATLAB path setup",
            "CircuitPilot MATLAB function (model generation and simulation)",
            "simulation result retrieval",
            "MATLAB Engine shutdown",
        }
        phase = match.group(1)
        return (phase if phase in known_phases else None), match.group(2)
    return None, type(exc).__name__


def _error_summary(exc: BaseException, stage: str) -> dict[str, Any]:
    phase, exception_type = _safe_phase(exc)
    failure_type = getattr(exc, "failure_type", None)
    if failure_type not in {"provider", "timeout", "unexpected", "planner", "simulation"}:
        failure_type = "timeout" if "timeout" in exception_type.lower() else stage
    category = getattr(exc, "error_category", failure_type)
    if category not in {"provider", "timeout", "unexpected", "planner", "simulation", "rate_limit"}:
        category = failure_type
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, bool) or not isinstance(status_code, int):
        status_code = None
    return {
        "category": str(category).upper(),
        "exception_type": exception_type,
        "phase": phase or stage,
        "http_status": status_code,
    }


def _clean(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _criteria_markdown(verification: dict[str, Any] | None) -> list[str]:
    if not verification:
        return ["Verification results are not available yet."]
    lines = ["| Check | Result | Measured value | Criterion | Explanation |", "|---|---|---:|---|---|"]
    checks = verification.get("checks")
    if not isinstance(checks, dict) or not checks:
        lines.append("| — | INCONCLUSIVE | — | — | No verification checks were recorded. |")
    else:
        for name, check in checks.items():
            if not isinstance(check, dict):
                continue
            value = check.get("value")
            unit = check.get("unit")
            measured = f"{value}{f' {unit}' if unit else ''}" if value is not None else "—"
            lines.append(
                f"| {_clean(name)} | {_status(check.get('status'))} | {_clean(measured)} | "
                f"{_clean(check.get('criterion') or 'No limit specified')} | {_clean(check.get('reason'))} |"
            )
    if verification.get("summary"):
        lines.extend(["", f"Verification summary: {_clean(verification['summary'])}"])
    return lines


def render_markdown(data: dict[str, Any]) -> str:
    verification = data.get("verification")
    repair = data.get("repair") or {}
    lines = [
        "# CircuitPilot Verification Report",
        "",
        f"- Run ID: `{_clean(data.get('run_id'))}`",
        f"- Timestamp (UTC): {_clean(data.get('timestamp_utc'))}",
        f"- Input: `{_clean(data.get('input_path'))}`",
        f"- Overall status: **{_overall_status(data.get('overall_status'))}**",
        f"- Run phase: **{_clean(data.get('lifecycle_status') or 'unknown')}**",
        f"- Current stage: {_clean(data.get('stage'))}",
        "",
        "## Generation and simulation",
        "",
        f"- Model generation: {_clean(data.get('model_generation_status'))}",
        f"- MATLAB startup: {_clean(data.get('matlab_startup_status'))}",
        f"- Diagram update: {_clean(data.get('diagram_update_status'))}",
        f"- Simulation: {_clean(data.get('simulation_status'))}",
        f"- Model path: `{_clean(data.get('model_path'))}`",
        f"- Samples: {_clean(data.get('sample_count'))}",
        f"- Simulation time range: {_clean(data.get('simulation_time_range'))}",
        "",
        "## Verification criteria and measurements",
        "",
        *_criteria_markdown(verification),
        "",
    ]
    if repair.get("enabled"):
        lines.extend([
            "## Automatic repair",
            "",
            f"- Repair attempts: {_clean(repair.get('attempt_count', 0))} of {_clean(repair.get('max_attempts'))}",
            f"- Final repair outcome: {_clean(repair.get('outcome'))}",
            "",
        ])
        for attempt in repair.get("attempts", []):
            lines.append(f"### Attempt {_clean(attempt.get('number'))} · {_clean(attempt.get('kind'))}")
            lines.append("")
            lines.append(f"- Status: **{_clean(attempt.get('status'))}**")
            lines.append(f"- Changes: {_clean(attempt.get('changes_summary'))}")
            if attempt.get("failure_reason"):
                lines.append(f"- Outcome detail: {_clean(attempt['failure_reason'])}")
            attempt_verification = attempt.get("verification") or {}
            for name, check in (attempt_verification.get("checks") or {}).items():
                value = check.get("value")
                unit = check.get("unit")
                measured = f"{value} {unit}" if value is not None and unit else str(value) if value is not None else "—"
                lines.append(f"- {_clean(name)}: **{_clean(check.get('status'))}**; {_clean(measured)}; {_clean(check.get('criterion') or 'No limit specified')}")
            lines.append("")
    error = data.get("error")
    if error:
        lines.extend([
            "## Error classification",
            "",
            f"- Category: {_clean(error.get('category'))}",
            f"- Stage: {_clean(error.get('phase'))}",
            f"- Exception type: {_clean(error.get('exception_type'))}",
            f"- HTTP status: {_clean(error.get('http_status'))}",
            "",
        ])
    lines.extend(["## Next useful step", "", _clean(data.get("next_step")), ""])
    lines.extend([
        "## Report limitations",
        "",
        "This report is updated at application checkpoints. Forced process termination or a machine crash can prevent the final update; the last saved report then represents the most recent checkpoint.",
        "",
    ])
    return "\n".join(lines)


class RunReportWriter:
    """Unique report directory for a non-repair CLI run."""

    def __init__(self, report_dir: str | Path, *, run_id: str, input_path: str):
        self.directory = Path(report_dir)
        self.directory.mkdir(parents=True, exist_ok=False)
        self._lock = threading.RLock()
        self.data: dict[str, Any] = {
            "run_id": run_id,
            "timestamp_utc": _utc_now(),
            "input_path": input_path,
            "overall_status": "inconclusive",
            "lifecycle_status": "running",
            "stage": "initializing",
            "model_generation_status": "not started",
            "matlab_startup_status": "not started",
            "diagram_update_status": "not started",
            "simulation_status": "not started",
            "model_path": None,
            "sample_count": None,
            "simulation_time_range": None,
            "verification": None,
            "repair": {"enabled": False},
            "error": None,
            "next_step": "The run has not reached a diagnostic checkpoint yet.",
        }
        self.write()

    @classmethod
    def resume(cls, report_dir: str | Path) -> "RunReportWriter":
        """Reopen a report checkpoint without creating or replacing its directory."""
        directory = Path(report_dir)
        data = json.loads((directory / "verification_report.json").read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not data.get("run_id"):
            raise ValueError("existing report checkpoint is invalid")
        writer = cls.__new__(cls)
        writer.directory = directory
        writer._lock = threading.RLock()
        writer.data = data
        return writer

    @classmethod
    def recover(cls, report_dir: str | Path, *, run_id: str, input_path: str) -> "RunReportWriter":
        """Create a replacement checkpoint in this run's directory if its report is unreadable."""
        writer = cls.__new__(cls)
        writer.directory = Path(report_dir)
        writer.directory.mkdir(parents=True, exist_ok=True)
        writer._lock = threading.RLock()
        writer.data = {
            "run_id": run_id, "timestamp_utc": _utc_now(), "input_path": input_path,
            "overall_status": "error", "lifecycle_status": "error", "stage": "recovery",
            "model_generation_status": "unknown", "matlab_startup_status": "unknown",
            "diagram_update_status": "unknown", "simulation_status": "unknown",
            "model_path": None, "sample_count": None, "simulation_time_range": None,
            "verification": None, "repair": {"enabled": False}, "error": None,
            "next_step": "Inspect the saved run artifacts; its previous report checkpoint could not be read.",
        }
        writer.write()
        return writer

    def update(self, **values: Any) -> None:
        with self._lock:
            self.data.update(values)
            self.write()

    def set_error(self, exc: BaseException, *, stage: str, interrupted: bool = False) -> None:
        with self._lock:
            error = _error_summary(exc, stage)
            if interrupted:
                error["category"] = "INTERRUPTED"
            if error["phase"] == "MATLAB Engine startup":
                self.data["matlab_startup_status"] = "error"
                self.data["model_generation_status"] = "error"
            elif error["phase"] == "MATLAB path setup":
                self.data["matlab_startup_status"] = "succeeded"
                self.data["diagram_update_status"] = "error"
                self.data["model_generation_status"] = "error"
            elif "MATLAB function" in str(error["phase"]):
                self.data["matlab_startup_status"] = "succeeded"
                self.data["diagram_update_status"] = "error"
                self.data["simulation_status"] = "error"
                self.data["model_generation_status"] = "error"
            self.update(
                overall_status="error",
                lifecycle_status="interrupted" if interrupted else "error",
                stage=stage,
                error=error,
                next_step=_next_step("error", error),
            )

    def set_result(self, result: dict[str, Any], verification: dict[str, Any]) -> None:
        with self._lock:
            self._set_result_locked(result, verification)

    def _set_result_locked(self, result: dict[str, Any], verification: dict[str, Any]) -> None:
        times = result.get("time")
        outputs = result.get("output")
        sample_count = len(outputs) if isinstance(outputs, (list, tuple)) else None
        time_range = None
        if isinstance(times, (list, tuple)) and times:
            time_range = f"{times[0]} to {times[-1]} s"
        startup = (
            "succeeded" if result.get("model_opened") is True or result.get("simulation_status") == "succeeded"
            else "failed" if result.get("model_opened") is False
            else "not reported"
        )
        self.update(
            overall_status=verification.get("status", "inconclusive"),
            lifecycle_status="completed",
            stage="verification complete",
            model_generation_status="succeeded" if result.get("model_path") else "unknown",
            matlab_startup_status=startup,
            diagram_update_status=str(result.get("diagram_update", "not reported")),
            simulation_status=str(result.get("simulation_status", "not reported")),
            model_path=result.get("model_path"),
            sample_count=sample_count,
            simulation_time_range=time_range,
            verification=verification,
            next_step=_next_step(verification.get("status", "inconclusive"), None),
        )

    def write(self) -> None:
        with self._lock:
            _write_report_file(self.directory / "verification_report.json", json.dumps(self.data, indent=2, ensure_ascii=False))
            _write_report_file(self.directory / "verification_report.md", render_markdown(self.data))


def _next_step(status: str, error: dict[str, Any] | None) -> str:
    if status == "fail":
        return "Review the failed checks and measured values, then adjust the structured model parameters or simulation settings before rerunning."
    if status == "pass":
        return "The specified checks passed. Review the saved model and simulation assumptions before using the result."
    if error:
        if error.get("category") == "TIMEOUT":
            return "The configured API execution deadline expired. Check the last report checkpoint and MATLAB/provider process; a Python timeout cannot forcibly stop a running MATLAB Engine call."
        phase = str(error.get("phase", ""))
        if "MATLAB Engine startup" in phase:
            return "Check that MATLAB and the matching MATLAB Engine for Python can start in this environment, then rerun."
        if "MATLAB" in phase or "simulation" in phase.lower():
            return "Inspect MATLAB/Simulink diagnostics for the recorded phase and verify the generated model before rerunning."
        if error.get("category") == "RATE_LIMIT":
            return "The provider rate-limited this request. Wait for quota recovery or adjust the provider quota, then retry within CircuitPilot's configured retry limit."
        if error.get("http_status") is not None or error.get("category") == "PROVIDER":
            return "Check provider availability, model access, and configured credentials without sharing credential values."
        return "Inspect the recorded stage and exception category, then correct the input or environment before rerunning."
    return "Review the missing criteria or simulation data; no pass or failure is inferred without sufficient evidence."


def write_repair_markdown(
    run_dir: str | Path,
    report: dict[str, Any],
    *,
    input_path: str | None,
    timestamp_utc: str,
) -> None:
    """Render a repair-loop checkpoint beside repair_report.json."""
    attempts = report.get("attempts") if isinstance(report.get("attempts"), list) else []
    pending = next((a for a in reversed(attempts) if a.get("status") in {"planning", "simulating"}), None)
    completed = next((a for a in reversed(attempts) if a.get("verification")), None)
    verification = completed.get("verification") if completed else None
    simulation: dict[str, Any] = {}
    if completed:
        path = Path(completed.get("attempt_dir", "")) / "simulation.json"
        if path.is_file():
            try:
                parsed = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    simulation = parsed
            except (OSError, json.JSONDecodeError):
                pass
    failure_phase = simulation.get("error_phase")
    if failure_phase == "MATLAB Engine startup":
        matlab_startup_status = "error"
    elif failure_phase:
        matlab_startup_status = "succeeded" if str(failure_phase) != "MATLAB Engine startup" else "error"
    else:
        matlab_startup_status = "succeeded" if simulation.get("model_opened") is True else "not reported"
    if failure_phase == "MATLAB path setup":
        diagram_status = "error"
    else:
        diagram_status = simulation.get("diagram_update", "not reported")
    result = {
        "run_id": report.get("run_id") or Path(run_dir).name,
        "timestamp_utc": timestamp_utc,
        "input_path": input_path or "Model specification (path unavailable)",
        "overall_status": report.get("status", "running"),
        "lifecycle_status": report.get("status", "running"),
        "stage": pending.get("status") if pending else "repair loop checkpoint",
        "model_generation_status": "succeeded" if completed and completed.get("model_path") else "not completed",
        "matlab_startup_status": matlab_startup_status,
        "diagram_update_status": diagram_status,
        "simulation_status": simulation.get("simulation_status", "not reported"),
        "model_path": completed.get("model_path") if completed else None,
        "sample_count": len(simulation.get("output", [])) if isinstance(simulation.get("output"), list) else None,
        "simulation_time_range": (
            f"{simulation['time'][0]} to {simulation['time'][-1]} s"
            if isinstance(simulation.get("time"), list) and simulation["time"] else None
        ),
        "verification": verification,
        "repair": {
            "enabled": True,
            "attempt_count": sum(1 for attempt in attempts if attempt.get("kind") == "repair"),
            "max_attempts": report.get("max_repair_attempts"),
            "outcome": report.get("status", "running"),
        },
        "error": None,
        "next_step": _repair_next_step(report, pending),
    }
    markdown = render_markdown(result)
    attempt_sections: list[str] = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        attempt_sections.extend([
            f"## Attempt {attempt.get('number', '?')} — {attempt.get('kind', 'unknown')}",
            "",
            f"- Status: **{_status(attempt.get('status'))}**",
            f"- Proposal validation: **{_status(attempt.get('proposal_validation', 'not_applicable'))}**",
            f"- Simulation execution: **{_status(attempt.get('simulation_execution_status', 'unknown'))}**",
            f"- Engineering verification: **{_status(attempt.get('engineering_verification_status', 'not_run'))}**",
            f"- Model: `{_clean(attempt.get('model_path'))}`",
            f"- Failure category: {_clean(attempt.get('failure_type'))}",
            f"- Explanation: {_clean(attempt.get('failure_reason'))}",
            "",
            *_criteria_markdown(attempt.get("verification")),
            "",
        ])
    if attempt_sections:
        markdown += "\n" + "\n".join(attempt_sections)
    _write_report_file(Path(run_dir) / "verification_report.md", markdown)


def _repair_next_step(report: dict[str, Any], pending: dict[str, Any] | None) -> str:
    status = report.get("status")
    if status == "running" and pending:
        return f"The run is currently {pending.get('status')}; inspect this attempt directory and wait for the call to finish before starting another run."
    if status == "pass":
        return "The saved verification criteria passed. Review the selected model and assumptions before use."
    if status == "fail":
        return "Review the failed verification measurements and remaining criteria; the repair limit is exhausted or a proposal was rejected."
    if status == "error":
        return "Review the sanitized failure category and attempt checkpoint; correct the provider, MATLAB, or input issue before another run."
    if status == "interrupted":
        return "The run was interrupted and checkpointed. Review the last attempt before deciding whether to resume with a new run."
    return "Review inconclusive checks and specify acceptance criteria if performance limits are required."
