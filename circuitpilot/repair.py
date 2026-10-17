"""Bounded, verification-driven repair orchestration for structured model specs."""

from __future__ import annotations

import json
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .ir import ModelSpec, validate_model
from .run_reports import write_repair_markdown
from .verification import VerificationReport, verify_simulation


class RepairProposal(BaseModel):
    """A bounded parameter/duration patch; it cannot execute code or replace topology."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1)
    changes_summary: str = Field(min_length=1)
    parameter_updates: dict[str, dict[str, Any]] = Field(default_factory=dict)
    stop_time: float | None = Field(default=None, gt=0, le=86400)

    @model_validator(mode="after")
    def has_changes(self) -> "RepairProposal":
        if not self.parameter_updates and self.stop_time is None:
            raise ValueError("A repair proposal must contain at least one change")
        if self.stop_time is not None and not math.isfinite(self.stop_time):
            raise ValueError("stop_time must be finite")
        if any(not updates for updates in self.parameter_updates.values()):
            raise ValueError("Each parameter update must change at least one parameter")
        return self


class FailedCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: float | str | None
    unit: str | None
    criterion: str | None
    reason: str


class RepairDiagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    failed_checks: list[FailedCheck] = Field(min_length=1)


class RepairPlanner(Protocol):
    def propose_repair(self, spec: ModelSpec, diagnosis: RepairDiagnosis) -> RepairProposal:
        """Return a targeted structured patch for the existing model."""


class RepairAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: int
    kind: Literal["original", "repair"]
    status: Literal["pass", "fail", "inconclusive", "rejected", "error", "planning", "simulating", "interrupted"]
    attempt_dir: str
    specification_path: str | None = None
    model_path: str | None = None
    changes_summary: str
    diagnosis: RepairDiagnosis | None = None
    verification: dict[str, Any] | None = None
    failure_reason: str | None = None
    retry_count: int | None = None
    provider_error: str | None = None
    transient_failure: bool | None = None
    failure_type: Literal["provider", "timeout", "planner", "unexpected", "simulation", "interrupted"] | None = None
    proposal_validation: Literal["not_applicable", "pending", "accepted", "rejected"] = "not_applicable"
    simulation_execution_status: Literal["not_started", "running", "succeeded", "failed", "timeout", "interrupted", "unknown"] = "not_started"
    engineering_verification_status: Literal["not_run", "pass", "fail", "inconclusive", "error"] = "not_run"


class RepairLoopReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["pass", "fail", "inconclusive", "error", "running", "interrupted"]
    run_dir: str
    max_repair_attempts: int
    best_attempt: int | None
    attempts: list[RepairAttempt]
    summary: str
    run_id: str | None = None
    timestamp_utc: str | None = None
    input_path: str | None = None


class RepairRejected(ValueError):
    """A repair patch was unsafe or failed the normal model validation boundary."""


def apply_repair(spec: ModelSpec, proposal: RepairProposal) -> ModelSpec:
    """Apply only PID tuning and bounded duration changes to the existing graph."""
    data = spec.model_dump(mode="python")
    components = {component["id"]: component for component in data["components"]}
    for component_id, updates in proposal.parameter_updates.items():
        if component_id not in components:
            raise RepairRejected(f"Repair references unknown component '{component_id}'.")
        component = components[component_id]
        if component["type"] != "PID Controller":
            raise RepairRejected("Repairs may change parameters only on an existing PID Controller.")
        unsupported = set(updates) - {"P", "I", "D", "N"}
        if unsupported:
            raise RepairRejected(f"Unsupported repair parameter(s): {', '.join(sorted(unsupported))}.")
        components[component_id]["parameters"].update(updates)
    if proposal.stop_time is not None:
        if proposal.stop_time > 86400:
            raise RepairRejected("Repair stop_time must not exceed 86400 seconds.")
        data["stop_time"] = proposal.stop_time
    try:
        repaired = ModelSpec.model_validate(data)
        validate_model(repaired)
    except Exception as exc:
        raise RepairRejected(f"Repair failed model/component/graph validation: {exc}") from exc
    return repaired


def diagnose(report: VerificationReport) -> RepairDiagnosis | None:
    """Include only failed checks; inconclusive metrics never trigger repair."""
    failed = [
        FailedCheck(
            name=name,
            value=check.value,
            unit=check.unit,
            criterion=check.criterion,
            reason=check.reason,
        )
        for name, check in report.checks.items()
        if check.status == "fail"
    ]
    if not failed:
        return None
    return RepairDiagnosis(summary=report.summary, failed_checks=failed)


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {str(key): _safe_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_safe_json_value(value), indent=2, allow_nan=False), encoding="utf-8")


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _rank(status: str) -> int:
    return {"fail": 0, "inconclusive": 1, "pass": 2}.get(status, -1)


def _quality(report: VerificationReport) -> tuple[int, int, int]:
    passed = sum(check.status == "pass" for check in report.checks.values())
    failed = sum(check.status == "fail" for check in report.checks.values())
    return (_rank(report.status), passed, -failed)


def _validated_verification(spec: ModelSpec, result: Any) -> tuple[VerificationReport | None, str | None]:
    """Treat a malformed verifier result as an error, never as a repair signal."""
    try:
        return VerificationReport.model_validate(verify_simulation(spec, result)), None
    except Exception as exc:
        return None, f"Verification produced an invalid report ({type(exc).__name__})."


def _write_checkpoint(
    run_dir: Path,
    *,
    status: Literal["running", "interrupted"],
    max_repair_attempts: int,
    attempts: list[RepairAttempt],
    best_attempt: int | None,
    summary: str,
    run_id: str,
    timestamp_utc: str,
    input_path: str | None,
    pending: RepairAttempt | None = None,
) -> None:
    snapshot_attempts = [*attempts, pending] if pending is not None else attempts
    report = RepairLoopReport(
        status=status,
        run_dir=str(run_dir),
        max_repair_attempts=max_repair_attempts,
        best_attempt=best_attempt,
        attempts=snapshot_attempts,
        summary=summary,
        run_id=run_id,
        timestamp_utc=timestamp_utc,
        input_path=input_path,
    )
    _write_json(run_dir / "repair_report.json", report.model_dump(mode="json"))
    write_repair_markdown(
        run_dir, report.model_dump(mode="json"), input_path=input_path,
        timestamp_utc=timestamp_utc,
    )


def _safe_failure_summary(exc: BaseException, *, stage: str) -> tuple[str, str, int | None]:
    """Return a report-safe failure category without exception text or payloads."""
    exception_type = type(exc).__name__
    if isinstance(exc, KeyboardInterrupt):
        return "The run was interrupted by the user.", "interrupted", None
    failure_type = getattr(exc, "failure_type", None)
    if failure_type not in {"provider", "timeout", "planner", "unexpected", "simulation", "interrupted"}:
        if "timeout" in exception_type.lower():
            failure_type = "timeout"
        elif stage == "planner":
            failure_type = "planner" if isinstance(exc, ValueError) else "unexpected"
        else:
            failure_type = stage
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, bool) or not isinstance(status_code, int):
        status_code = None
    if failure_type == "timeout":
        reason = "The operation timed out."
    elif failure_type == "provider":
        reason = f"The provider request failed{f' with HTTP {status_code}' if status_code is not None else ''}."
    elif failure_type == "simulation":
        if "MATLAB failed to start" in str(exc):
            reason = f"MATLAB failed to start ({exception_type})."
        else:
            reason = f"Model generation or MATLAB simulation failed ({exception_type})."
    elif failure_type == "interrupted":
        reason = "The run was interrupted by the user."
    else:
        reason = f"The {stage} operation failed ({exception_type})."
    return reason, failure_type, status_code


def _safe_failure_phase(exc: BaseException, default: str) -> str:
    match = re.match(r"^(.*?) failed \(([A-Za-z][A-Za-z0-9_.]*)\):", str(exc))
    known_phases = {
        "MATLAB Engine startup",
        "MATLAB path setup",
        "CircuitPilot MATLAB function (model generation and simulation)",
        "simulation result retrieval",
        "MATLAB Engine shutdown",
    }
    phase = match.group(1) if match else None
    return phase if phase in known_phases else default


def run_repair_loop(
    spec: ModelSpec,
    planner: RepairPlanner,
    generator: Any,
    output_dir: str | Path,
    *,
    max_repair_attempts: int = 2,
    run_id: str | None = None,
    input_path: str | None = None,
    on_stage: Any = None,
    should_stop: Any = None,
) -> RepairLoopReport:
    """Run one baseline plus at most N validated repairs, isolating every artifact."""
    if (isinstance(max_repair_attempts, bool) or not isinstance(max_repair_attempts, int)
            or not 0 <= max_repair_attempts <= 5):
        raise ValueError("max_repair_attempts must be an integer between 0 and 5")
    validate_model(spec)
    identifier = run_id or _new_run_id()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
        raise ValueError("run_id may contain only letters, digits, underscores, and hyphens")
    run_dir = Path(output_dir).expanduser().resolve() / "repairs" / identifier
    run_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    interrupted = False

    attempts: list[RepairAttempt] = []
    current = spec
    best_attempt: int | None = None
    best_quality: tuple[int, int, int] = (-1, -1, -1)
    final_status: Literal["pass", "fail", "inconclusive", "error", "interrupted"] = "error"
    summary = "Repair orchestration did not complete."

    # This file is written before any MATLAB or provider call. Subsequent checkpoints
    # make abrupt process termination distinguishable from a completed repair run.
    _write_checkpoint(
        run_dir, status="running", max_repair_attempts=max_repair_attempts,
        attempts=attempts, best_attempt=best_attempt, summary="Repair run initialized.",
        run_id=identifier, timestamp_utc=started_at, input_path=input_path,
    )

    def stage(name: str) -> None:
        if should_stop is not None and should_stop():
            raise TimeoutError("CircuitPilot API execution deadline exceeded")
        if on_stage is not None:
            on_stage(name)

    for number in range(max_repair_attempts + 1):
        kind: Literal["original", "repair"] = "original" if number == 0 else "repair"
        attempt_dir = run_dir / f"attempt_{number:02d}_{kind}"
        attempt_dir.mkdir()
        spec_path = attempt_dir / "spec.json"
        _write_json(spec_path, current.model_dump(mode="json"))
        proposal: RepairProposal | None = None
        diagnosis: RepairDiagnosis | None = None
        changes = "Original user specification." if number == 0 else ""

        if number > 0:
            prior = attempts[-1]
            try:
                prior_report = VerificationReport.model_validate(prior.verification)
            except Exception as exc:
                reason = f"Prior verification report was invalid ({type(exc).__name__}); repair stopped safely."
                attempts.append(RepairAttempt(
                    number=number, kind=kind, status="error", attempt_dir=str(attempt_dir),
                    specification_path=str(spec_path), changes_summary="No repair proposal was requested.",
                    failure_reason=reason, failure_type="unexpected",
                    engineering_verification_status="error",
                ))
                _write_json(attempt_dir / "failure.json", {"failure_type": "unexpected", "failure_reason": reason})
                final_status, summary = "error", reason
                break
            diagnosis = diagnose(prior_report)
            if diagnosis is None:
                # Defensive: only failed checks are sent to the repair planner.
                final_status = prior_report.status
                summary = "No failed verification checks remain; no repair was requested."
                attempt_dir.rmdir()
                break
            pending = RepairAttempt(
                number=number, kind=kind, status="planning", attempt_dir=str(attempt_dir),
                specification_path=str(spec_path), changes_summary="Awaiting structured repair proposal.",
                diagnosis=diagnosis, proposal_validation="pending",
            )
            _write_checkpoint(
                run_dir, status="running", max_repair_attempts=max_repair_attempts,
                attempts=attempts, best_attempt=best_attempt,
                summary="Repair planner request in progress.", pending=pending,
                run_id=identifier, timestamp_utc=started_at, input_path=input_path,
            )
            try:
                stage("planning")
                proposal = planner.propose_repair(current, diagnosis)
                stage("validation")
            except KeyboardInterrupt as exc:
                reason, failure_type, _ = _safe_failure_summary(exc, stage="planner")
                attempts.append(RepairAttempt(
                    number=number, kind=kind, status="interrupted", attempt_dir=str(attempt_dir),
                    specification_path=str(spec_path), changes_summary="No repair proposal was received.",
                    diagnosis=diagnosis, failure_reason=reason, retry_count=0,
                    failure_type="interrupted" if failure_type != "timeout" else "timeout",
                ))
                _write_json(attempt_dir / "failure.json", {
                    "failure_type": "interrupted", "failure_reason": reason,
                })
                final_status, summary = "interrupted", reason
                interrupted = True
                break
            except Exception as exc:
                provider_error, failure_type, status_code = _safe_failure_summary(exc, stage="planner")
                retry_count = getattr(exc, "retry_count", 0)
                transient_failure = getattr(exc, "transient", None)
                if failure_type not in {"provider", "timeout", "planner", "unexpected"}:
                    failure_type = "unexpected"
                if not isinstance(retry_count, int) or isinstance(retry_count, bool) or retry_count < 0:
                    retry_count = 0
                if transient_failure is not True and transient_failure is not False and transient_failure is not None:
                    transient_failure = None
                reason = f"Repair planning failed: {provider_error}"
                attempts.append(RepairAttempt(
                    number=number, kind=kind, status="error", attempt_dir=str(attempt_dir),
                    specification_path=str(spec_path), changes_summary="No repair proposal was received.",
                    diagnosis=diagnosis, failure_reason=reason,
                    retry_count=retry_count, provider_error=provider_error,
                    transient_failure=transient_failure,
                    failure_type=failure_type,
                ))
                _write_json(attempt_dir / "failure.json", {
                    "failure_type": failure_type,
                    "failure_reason": reason,
                    "retry_count": retry_count,
                    "provider_error": provider_error,
                    "transient_failure": transient_failure,
                    "http_status": status_code,
                })
                final_status, summary = "error", reason
                break
            try:
                proposal = RepairProposal.model_validate(proposal)
                changes = proposal.changes_summary
                _write_json(attempt_dir / "proposal.json", proposal.model_dump(mode="json"))
                current = apply_repair(current, proposal)
                _write_json(spec_path, current.model_dump(mode="json"))
            except Exception as exc:
                reason = (
                    f"Repair validation failed ({type(exc).__name__})."
                    if type(exc).__name__ == "RepairRejected"
                    else f"Repair proposal rejected ({type(exc).__name__})."
                )
                attempts.append(RepairAttempt(
                    number=number, kind=kind, status="rejected", attempt_dir=str(attempt_dir),
                    specification_path=str(spec_path) if spec_path.exists() else None,
                    changes_summary=changes or "No validated changes were applied.",
                    diagnosis=diagnosis, failure_reason=reason,
                    failure_type="planner", proposal_validation="rejected",
                ))
                _write_json(attempt_dir / "failure.json", {
                    "failure_type": "planner", "failure_reason": reason,
                })
                final_status, summary = "fail", reason
                break

        pending_simulation = RepairAttempt(
            number=number, kind=kind, status="simulating", attempt_dir=str(attempt_dir),
            specification_path=str(spec_path), changes_summary=changes, diagnosis=diagnosis,
            proposal_validation="accepted" if number else "not_applicable",
            simulation_execution_status="running",
        )
        _write_checkpoint(
            run_dir, status="running", max_repair_attempts=max_repair_attempts,
            attempts=attempts, best_attempt=best_attempt,
            summary="Simulation in progress.", pending=pending_simulation,
            run_id=identifier, timestamp_utc=started_at, input_path=input_path,
        )
        try:
            stage("generation")
            stage("simulation")
            result = generator(current, attempt_dir)
        except KeyboardInterrupt as exc:
            reason, _, _ = _safe_failure_summary(exc, stage="simulation")
            attempts.append(RepairAttempt(
                number=number, kind=kind, status="interrupted", attempt_dir=str(attempt_dir),
                specification_path=str(spec_path), changes_summary=changes,
                diagnosis=diagnosis, failure_reason=reason, failure_type="interrupted",
                proposal_validation="accepted" if number else "not_applicable",
                simulation_execution_status="interrupted",
            ))
            _write_json(attempt_dir / "failure.json", {
                "failure_type": "interrupted", "failure_reason": reason,
            })
            final_status, summary = "interrupted", reason
            interrupted = True
            break
        except Exception as exc:
            # MATLAB/generator exceptions are recorded and terminate safely.
            reason, failure_type, status_code = _safe_failure_summary(exc, stage="simulation")
            result = {
                "simulation_status": "failed",
                "error_category": failure_type,
                "error_type": type(exc).__name__,
                "http_status": status_code,
                "error_phase": _safe_failure_phase(exc, "model generation or simulation"),
            }
            verification, verification_error = _validated_verification(current, result)
            if verification is None:
                reason = f"Generation or MATLAB simulation failed: {reason}; {verification_error}"
                _write_json(attempt_dir / "simulation.json", result)
                attempts.append(RepairAttempt(
                    number=number, kind=kind, status="error", attempt_dir=str(attempt_dir),
                    specification_path=str(spec_path), changes_summary=changes,
                    failure_reason=reason, failure_type="unexpected",
                    proposal_validation="accepted" if number else "not_applicable",
                    simulation_execution_status="failed", engineering_verification_status="error",
                ))
                _write_json(attempt_dir / "failure.json", {"failure_type": "unexpected", "failure_reason": reason})
                final_status, summary = "error", reason
                break
            reason = f"Generation or MATLAB simulation failed: {reason}"
            _write_json(attempt_dir / "simulation.json", result)
            _write_json(attempt_dir / "verification.json", verification.model_dump(mode="json"))
            attempts.append(RepairAttempt(
                number=number, kind=kind, status="error", attempt_dir=str(attempt_dir),
                specification_path=str(spec_path),
                model_path=str(result.get("model_path")) if result.get("model_path") else None,
                changes_summary=changes, diagnosis=diagnosis,
                verification=verification.model_dump(mode="json"), failure_reason=reason,
                failure_type=failure_type if failure_type in {"timeout", "interrupted", "simulation", "unexpected"} else "simulation",
                proposal_validation="accepted" if number else "not_applicable",
                simulation_execution_status="timeout" if failure_type == "timeout" else "failed",
                engineering_verification_status=verification.status,
            ))
            final_status, summary = "error", reason
            break

        if not isinstance(result, dict):
            reason = "Generator returned an invalid result; repair stopped safely."
            attempts.append(RepairAttempt(
                number=number, kind=kind, status="error", attempt_dir=str(attempt_dir),
                specification_path=str(spec_path), changes_summary=changes,
                failure_reason=reason, failure_type="unexpected",
                proposal_validation="accepted" if number else "not_applicable",
                simulation_execution_status="unknown", engineering_verification_status="error",
            ))
            _write_json(attempt_dir / "failure.json", {"failure_type": "unexpected", "failure_reason": reason})
            final_status, summary = "error", reason
            break

        execution_status = "succeeded" if result.get("simulation_status") == "succeeded" else "failed"
        if execution_status != "succeeded":
            verification, verification_error = _validated_verification(current, result)
            if verification is None:
                reason = verification_error or "Verification report was invalid."
                _write_json(attempt_dir / "simulation.json", result)
                attempts.append(RepairAttempt(
                    number=number, kind=kind, status="error", attempt_dir=str(attempt_dir),
                    specification_path=str(spec_path), model_path=result.get("model_path"),
                    changes_summary=changes, failure_reason=reason, failure_type="unexpected",
                    proposal_validation="accepted" if number else "not_applicable",
                    simulation_execution_status=execution_status, engineering_verification_status="error",
                ))
                _write_json(attempt_dir / "failure.json", {"failure_type": "unexpected", "failure_reason": reason})
                final_status, summary = "error", reason
                break
            _write_json(attempt_dir / "simulation.json", result)
            _write_json(attempt_dir / "verification.json", verification.model_dump(mode="json"))
            reason = "Generation or simulation did not complete successfully; repair stopped without another proposal."
            attempts.append(RepairAttempt(
                number=number, kind=kind, status="error", attempt_dir=str(attempt_dir),
                specification_path=str(spec_path), model_path=result.get("model_path"),
                changes_summary=changes, diagnosis=diagnosis,
                verification=verification.model_dump(mode="json"), failure_reason=reason,
                failure_type="simulation", proposal_validation="accepted" if number else "not_applicable",
                simulation_execution_status=execution_status, engineering_verification_status=verification.status,
            ))
            final_status, summary = "error", reason
            break

        verification, verification_error = _validated_verification(current, result)
        if verification is None:
            reason = verification_error or "Verification report was invalid."
            _write_json(attempt_dir / "simulation.json", result)
            attempts.append(RepairAttempt(
                number=number, kind=kind, status="error", attempt_dir=str(attempt_dir),
                specification_path=str(spec_path), model_path=result.get("model_path"),
                changes_summary=changes, failure_reason=reason, failure_type="unexpected",
                proposal_validation="accepted" if number else "not_applicable",
                simulation_execution_status=execution_status, engineering_verification_status="error",
            ))
            _write_json(attempt_dir / "failure.json", {"failure_type": "unexpected", "failure_reason": reason})
            final_status, summary = "error", reason
            break
        _write_json(attempt_dir / "simulation.json", result)
        _write_json(attempt_dir / "verification.json", verification.model_dump(mode="json"))
        failed = diagnose(verification)
        status = verification.status
        attempt = RepairAttempt(
            number=number, kind=kind, status=status, attempt_dir=str(attempt_dir),
            specification_path=str(spec_path),
            model_path=str(result.get("model_path")) if result.get("model_path") else None,
            changes_summary=changes, diagnosis=failed,
            verification=verification.model_dump(mode="json"),
            failure_reason=(
                "; ".join(
                    f"{check.name}={check.value!r} {check.unit or ''}: {check.reason}"
                    for check in failed.failed_checks
                )
                if failed else None
            ),
            proposal_validation="accepted" if number else "not_applicable",
            simulation_execution_status=execution_status,
            engineering_verification_status=verification.status,
        )
        attempts.append(attempt)
        quality = _quality(verification)
        if quality > best_quality:
            best_quality, best_attempt = quality, number

        _write_checkpoint(
            run_dir, status="running", max_repair_attempts=max_repair_attempts,
            attempts=attempts, best_attempt=best_attempt,
            summary="Simulation attempt recorded; repair loop continuing if needed.",
            run_id=identifier, timestamp_utc=started_at, input_path=input_path,
        )

        try:
            stage("verification")
        except Exception as exc:
            reason, failure_type, _ = _safe_failure_summary(exc, stage="verification")
            attempts[-1] = attempts[-1].model_copy(update={
                "status": "error", "failure_reason": reason,
                "failure_type": failure_type if failure_type in {"timeout", "interrupted"} else "unexpected",
            })
            final_status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
            summary = reason
            interrupted = isinstance(exc, KeyboardInterrupt)
            break
        if status == "pass":
            final_status, summary = "pass", "Simulation passed all specified acceptance criteria."
            break
        if failed is None:  # Includes inconclusive metrics: no repair trigger.
            final_status, summary = "inconclusive", "No failed checks require repair; at least one result is inconclusive."
            break
        if number == max_repair_attempts:
            final_status, summary = "fail", "Repair attempt limit reached with failed verification checks."
            break

    report = RepairLoopReport(
        status=final_status, run_dir=str(run_dir), max_repair_attempts=max_repair_attempts,
        best_attempt=best_attempt, attempts=attempts, summary=summary,
        run_id=identifier, timestamp_utc=started_at, input_path=input_path,
    )
    _write_json(run_dir / "repair_report.json", report.model_dump(mode="json"))
    write_repair_markdown(
        run_dir, report.model_dump(mode="json"), input_path=input_path,
        timestamp_utc=started_at,
    )
    if interrupted:
        raise KeyboardInterrupt
    return report
