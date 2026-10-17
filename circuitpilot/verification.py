"""Post-simulation checks for response integrity and user-defined criteria."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .ir import ModelSpec


VerificationStatus = Literal["pass", "fail", "inconclusive"]


class VerificationCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: VerificationStatus
    value: float | str | None = None
    unit: str | None = None
    criterion: str | None = None
    reason: str


class VerificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: VerificationStatus
    checks: dict[str, VerificationCheck]
    summary: str


def _check(
    status: VerificationStatus,
    reason: str,
    *,
    value: float | str | None = None,
    unit: str | None = None,
    criterion: str | None = None,
) -> VerificationCheck:
    return VerificationCheck(status=status, value=value, unit=unit, criterion=criterion, reason=reason)


def _numeric_series(value: Any) -> tuple[list[float] | None, str | None]:
    if not isinstance(value, (list, tuple)) or not value:
        return None, "No samples were returned."
    values: list[float] = []
    for sample in value:
        if isinstance(sample, bool) or not isinstance(sample, (int, float)):
            return None, "Samples must all be numeric values."
        numeric = float(sample)
        if not math.isfinite(numeric):
            return None, "One or more samples are NaN or infinite."
        values.append(numeric)
    return values, None


def _step_reference(spec: ModelSpec) -> tuple[float, float, float] | None:
    steps = [component for component in spec.components if component.type == "Step"]
    if len(steps) != 1:
        return None
    params = steps[0].parameters
    return float(params["Time"]), float(params["Before"]), float(params["After"])


def _inconclusive_metric(reason: str, *, unit: str | None = None) -> VerificationCheck:
    return _check("inconclusive", reason, unit=unit)


def verify_simulation(spec: ModelSpec, result: dict[str, Any]) -> VerificationReport:
    """Check generator output; performance limits only apply when specified by the spec.

    Step.After supplies a tracking target when exactly one Step exists. Users can
    override it with ``spec.verification.expected_final_value``. Unspecified limits
    leave metric checks inconclusive while retaining any measured value.
    """
    criteria = spec.verification
    checks: dict[str, VerificationCheck] = {}

    simulation_status = result.get("simulation_status")
    if simulation_status == "succeeded":
        checks["simulation_success"] = _check("pass", "The generator reported successful simulation.", value=simulation_status)
    elif simulation_status in {"failed", "error"}:
        checks["simulation_success"] = _check("fail", f"The generator reported simulation status '{simulation_status}'.", value=str(simulation_status))
    else:
        checks["simulation_success"] = _check("inconclusive", "The result does not contain a recognized successful simulation status.", value=str(simulation_status) if simulation_status is not None else None)

    outputs, output_error = _numeric_series(result.get("output"))
    if outputs is None:
        checks["response_finite"] = _check("fail", output_error or "Response samples are unavailable.")
    else:
        checks["response_finite"] = _check("pass", "All response samples are finite.", value=len(outputs), unit="samples")

    times, time_error = _numeric_series(result.get("time"))
    if times is None:
        checks["sample_data"] = _check("fail", f"Simulation time samples are invalid: {time_error}")
    elif outputs is None:
        checks["sample_data"] = _check("inconclusive", "Response samples are unavailable, so time and response alignment cannot be checked.")
    elif len(times) != len(outputs):
        checks["sample_data"] = _check("fail", f"Time and response lengths differ ({len(times)} vs {len(outputs)}).")
    elif len(times) < 2:
        checks["sample_data"] = _check("inconclusive", "At least two aligned samples are required for response metrics.", value=len(times), unit="samples")
    elif any(right <= left for left, right in zip(times, times[1:])):
        checks["sample_data"] = _check("fail", "Time samples must be strictly increasing.")
    else:
        checks["sample_data"] = _check("pass", "Time and response samples are aligned and strictly increasing.", value=len(times), unit="samples")

    usable = outputs is not None and times is not None and len(times) == len(outputs) and len(times) >= 2 and checks["sample_data"].status == "pass"
    step = _step_reference(spec)
    target = criteria.expected_final_value if criteria and criteria.expected_final_value is not None else (step[2] if step else None)

    reference_reached = step is None or (usable and times[-1] >= step[0])
    if usable and target is not None and reference_reached:
        final_error = abs(outputs[-1] - target)
        limit = criteria.max_tracking_error if criteria else None
        if limit is None:
            checks["final_value_tracking_error"] = _check(
                "inconclusive", "Tracking error was measured; no maximum tracking-error criterion was specified.",
                value=final_error, unit="output units", criterion=None,
            )
        elif final_error <= limit:
            checks["final_value_tracking_error"] = _check(
                "pass", "Final-value tracking error is within the specified limit.",
                value=final_error, unit="output units", criterion=f"<= {limit} output units",
            )
        else:
            checks["final_value_tracking_error"] = _check(
                "fail", "Final-value tracking error exceeds the specified limit.",
                value=final_error, unit="output units", criterion=f"<= {limit} output units",
            )
    elif usable:
        reason = (
            "Simulation ended before the Step reference transition."
            if step is not None and not reference_reached
            else "No expected final value was specified or inferable from exactly one Step block."
        )
        checks["final_value_tracking_error"] = _inconclusive_metric(reason, unit="output units")
    else:
        checks["final_value_tracking_error"] = _inconclusive_metric("Aligned response samples are insufficient.", unit="output units")

    if usable and step is not None and abs(step[2] - step[1]) > 0:
        step_time, baseline, step_target = step
        post_step = [(time, output) for time, output in zip(times, outputs) if time >= step_time]
        if not post_step:
            checks["peak_overshoot"] = _inconclusive_metric("No response samples occur after the Step transition.", unit="%")
            checks["settling_time"] = _inconclusive_metric("No response samples occur after the Step transition.", unit="s")
            post_step = []
        if post_step:
            peak = max((y - step_target) * (1.0 if step_target > baseline else -1.0) for _, y in post_step)
            overshoot = max(0.0, peak) / abs(step_target - baseline) * 100.0
            if not math.isfinite(overshoot):
                checks["peak_overshoot"] = _check("fail", "Peak overshoot calculation was not finite.", unit="%")
                overshoot = None
        else:
            overshoot = None
        limit = criteria.max_overshoot_percent if criteria else None
        if overshoot is None:
            pass
        elif limit is None:
            checks["peak_overshoot"] = _check(
                "inconclusive", "Peak overshoot was measured; no maximum overshoot criterion was specified.",
                value=overshoot, unit="%", criterion=None,
            )
        elif overshoot <= limit:
            checks["peak_overshoot"] = _check("pass", "Peak overshoot is within the specified limit.", value=overshoot, unit="%", criterion=f"<= {limit}%")
        else:
            checks["peak_overshoot"] = _check("fail", "Peak overshoot exceeds the specified limit.", value=overshoot, unit="%", criterion=f"<= {limit}%")

        if post_step:
            band_percent = criteria.settling_band_percent if criteria else 2.0
            band = abs(step_target - baseline) * band_percent / 100.0
            post_step_indices = [index for index, time in enumerate(times) if time >= step_time]
            outside_indices = [index for index in post_step_indices if abs(outputs[index] - step_target) > band]
            settle_index = (outside_indices[-1] + 1) if outside_indices else post_step_indices[0]
            if settle_index < len(times):
                settling_time = max(0.0, times[settle_index] - step_time)
                limit = criteria.max_settling_time if criteria else None
                if limit is None:
                    checks["settling_time"] = _check(
                        "inconclusive", f"Settling time uses a {band_percent:g}% step-amplitude band; no maximum settling-time criterion was specified.",
                        value=settling_time, unit="s", criterion=None,
                    )
                elif settling_time <= limit:
                    checks["settling_time"] = _check("pass", "Settling time is within the specified limit.", value=settling_time, unit="s", criterion=f"<= {limit} s; {band_percent:g}% band")
                else:
                    checks["settling_time"] = _check("fail", "Settling time exceeds the specified limit.", value=settling_time, unit="s", criterion=f"<= {limit} s; {band_percent:g}% band")
            else:
                status: VerificationStatus = "fail" if criteria and criteria.max_settling_time is not None else "inconclusive"
                reason = "The response did not enter and remain inside the settling band before simulation ended."
                checks["settling_time"] = _check(status, reason, criterion=(f"<= {criteria.max_settling_time} s" if criteria and criteria.max_settling_time is not None else None), unit="s")
    else:
        reason = "Settling and overshoot require aligned samples and exactly one nonzero Step transition."
        checks["peak_overshoot"] = _inconclusive_metric(reason, unit="%")
        checks["settling_time"] = _inconclusive_metric(reason, unit="s")

    statuses = [check.status for check in checks.values()]
    if "fail" in statuses:
        overall: VerificationStatus = "fail"
        summary = "One or more required simulation checks failed."
    elif "inconclusive" in statuses:
        overall = "inconclusive"
        summary = "Simulation integrity passed where checkable; at least one performance result is informational or lacks enough data/criteria."
    else:
        overall = "pass"
        summary = "Simulation integrity and all specified performance criteria passed."
    return VerificationReport(status=overall, checks=checks, summary=summary)
