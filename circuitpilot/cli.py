"""Command line entry point for deterministic model generation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .generator import generate_and_simulate
from .ir import ModelSpec, validate_model
from .planner import GeminiPlanner, OpenAIPlanner, PlannerError, PROVIDERS
from .repair import run_repair_loop
from .run_reports import RunReportWriter, new_run_id
from .verification import verify_simulation
from .service import run_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and simulate a Simulink model from CircuitPilot IR JSON.")
    parser.add_argument("spec", type=Path, nargs="?", help="Model specification JSON")
    parser.add_argument("--prompt", help="Natural-language classical control-system description")
    parser.add_argument("--graph-output", type=Path, help="Where to save the planned graph JSON")
    parser.add_argument(
        "--provider", choices=tuple(PROVIDERS),
        help="LLM provider (defaults to CIRCUITPILOT_PROVIDER or openai)",
    )
    parser.add_argument(
        "--model",
        help="Model name (defaults to CIRCUITPILOT_MODEL or the provider default)",
    )
    parser.add_argument(
        "--diagnose-gemini",
        action="store_true",
        help="Run a minimal Gemini structured-output probe before planning",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts"), help="Output directory for the .slx")
    parser.add_argument(
        "--auto-repair", action="store_true",
        help="Run bounded verification-driven structured repairs after failed checks",
    )
    parser.add_argument(
        "--max-repair-attempts", type=int, default=2,
        help="Maximum repair proposals after the original run (default: 2; requires --auto-repair)",
    )
    args = parser.parse_args()
    if (args.spec is None) == (args.prompt is None):
        parser.error("provide exactly one of a specification JSON path or --prompt")
    if args.max_repair_attempts < 0:
        parser.error("--max-repair-attempts must be nonnegative")
    run_id = new_run_id()
    input_path = str(args.spec.expanduser().resolve()) if args.spec else "Natural-language prompt (content omitted)"
    writer = RunReportWriter(args.output / "run_reports" / run_id, run_id=run_id, input_path=input_path)

    # Keep one shared baseline workflow for CLI and API callers. Diagnostic and
    # explicitly opted-in repair flows retain their specialized paths below.
    if not args.auto_repair and not args.diagnose_gemini:
        try:
            spec = ModelSpec.model_validate_json(args.spec.read_text(encoding="utf-8")) if args.spec else None
            provider = (args.provider or os.getenv("CIRCUITPILOT_PROVIDER", "openai")).strip().lower()
            planner = None
            if args.prompt is not None:
                planner = GeminiPlanner(model=args.model) if provider == "gemini" else OpenAIPlanner(provider=provider, model=args.model)
            result = run_baseline(
                output_dir=args.output,
                writer=writer,
                spec=spec,
                prompt=args.prompt,
                provider=provider,
                model=args.model,
                planner=planner,
                generator=generate_and_simulate,
                graph_output=args.graph_output or (args.output / "planned.graph.json" if args.prompt else None),
            )
        except KeyboardInterrupt:
            parser.exit(130, "Interrupted by user. Run report saved.\n")
        except Exception as exc:
            if isinstance(exc, PlannerError):
                parser.exit(2, f"Planning or validation error: {exc}\n")
            raise
        print(json.dumps(result, indent=2))
        return

    if args.prompt is not None:
        writer.update(stage="planning")
        try:
            provider = (args.provider or os.getenv("CIRCUITPILOT_PROVIDER", "openai")).strip().lower()
            if args.diagnose_gemini and provider != "gemini":
                writer.set_error(ValueError("Gemini diagnostic requires Gemini provider."), stage="Gemini diagnostic")
                parser.error("--diagnose-gemini requires the gemini provider")
            planner = GeminiPlanner(model=args.model) if provider == "gemini" else OpenAIPlanner(provider=provider, model=args.model)
            if args.diagnose_gemini:
                probes = planner.diagnose_api()
                print(json.dumps({
                    "provider": "gemini",
                    "model": planner.model,
                    "credential_source": planner.credential_source,
                    "credential_configured": planner.credential_configured,
                    "endpoint": planner.api_endpoint,
                    "probes": probes,
                }, indent=2))
                if not all(
                    probe["success"] for probe in probes
                    if probe["probe"].startswith("interactions_")
                ):
                    writer.update(
                        overall_status="error",
                        lifecycle_status="error",
                        stage="Gemini diagnostic",
                        error={
                            "category": "PROVIDER",
                            "exception_type": "GeminiDiagnosticError",
                            "phase": "Gemini diagnostic",
                            "http_status": None,
                        },
                        next_step="Review the sanitized probe results printed above, then verify provider endpoint, model access, and credentials without exposing credential values.",
                    )
                    parser.exit(1)
                writer.update(lifecycle_status="completed", stage="Gemini diagnostic complete")
                return
            spec = planner.plan(args.prompt)
        except KeyboardInterrupt as exc:
            writer.set_error(exc, stage="planning", interrupted=True)
            parser.exit(130, "Interrupted by user. Run report saved.\n")
        except Exception as exc:
            writer.set_error(exc, stage="planning")
            if isinstance(exc, PlannerError):
                parser.exit(2, f"Planning error: {exc}\n")
            raise
        graph_path = args.graph_output or (args.output / f"{spec.name}.graph.json")
        try:
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            graph_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        except Exception as exc:
            writer.set_error(exc, stage="graph save")
            raise
        print(f"Validated graph saved to {graph_path}", file=sys.stderr)
    else:
        writer.update(stage="specification loading")
        try:
            spec = ModelSpec.model_validate_json(args.spec.read_text(encoding="utf-8"))
        except KeyboardInterrupt as exc:
            writer.set_error(exc, stage="specification loading", interrupted=True)
            parser.exit(130, "Interrupted by user. Run report saved.\n")
        except Exception as exc:
            writer.set_error(exc, stage="specification loading")
            raise
    try:
        validate_model(spec)
    except Exception as exc:
        writer.set_error(exc, stage="specification validation")
        raise

    if args.auto_repair:
        writer.update(
            stage="repair initialization",
            repair={"enabled": True, "attempt_count": 0, "max_attempts": args.max_repair_attempts, "outcome": "running"},
        )
        try:
            provider = (args.provider or os.getenv("CIRCUITPILOT_PROVIDER", "openai")).strip().lower()
            repair_planner = (
                GeminiPlanner(model=args.model)
                if provider == "gemini"
                else OpenAIPlanner(provider=provider, model=args.model)
            )
            report = run_repair_loop(
                spec,
                repair_planner,
                generate_and_simulate,
                args.output,
                max_repair_attempts=args.max_repair_attempts,
                run_id=run_id,
                input_path=input_path,
            )
        except KeyboardInterrupt as exc:
            writer.set_error(exc, stage="repair", interrupted=True)
            parser.exit(130, "Interrupted by user. Run report saved.\n")
        except Exception as exc:
            writer.set_error(exc, stage="repair initialization")
            if isinstance(exc, (PlannerError, ValueError)):
                parser.exit(2, f"Repair orchestration error: {exc}\n")
            raise
        latest_attempt = next(
            (attempt for attempt in reversed(report.attempts) if attempt.verification is not None),
            None,
        )
        if latest_attempt is not None:
            simulation_path = Path(latest_attempt.attempt_dir) / "simulation.json"
            try:
                simulation_result = json.loads(simulation_path.read_text(encoding="utf-8"))
                writer.set_result(simulation_result, latest_attempt.verification)
            except (OSError, json.JSONDecodeError):
                writer.update(verification=latest_attempt.verification)
        error = None
        if report.status == "error" and report.attempts:
            failed = report.attempts[-1]
            error = {
                "category": (failed.failure_type or "unexpected").upper(),
                "exception_type": None,
                "phase": "repair planning" if failed.kind == "repair" else "model generation or simulation",
                "http_status": None,
            }
        writer.update(
            overall_status=report.status,
            lifecycle_status="completed",
            stage="repair complete",
            repair={
                "enabled": True,
                "attempt_count": sum(attempt.kind == "repair" for attempt in report.attempts),
                "max_attempts": report.max_repair_attempts,
                "outcome": report.status,
            },
            error=error,
            next_step=(report.summary if report.status not in {"error", "fail"} else "Review the saved attempt diagnosis and safe failure category before another run."),
        )
        print(report.model_dump_json(indent=2))
        return

    writer.update(
        stage="model generation and simulation",
        model_generation_status="in progress",
        matlab_startup_status="in progress",
        simulation_status="in progress",
    )
    try:
        simulation_result = generate_and_simulate(spec, args.output)
    except KeyboardInterrupt as exc:
        writer.set_error(exc, stage="model generation and simulation", interrupted=True)
        parser.exit(130, "Interrupted by user. Run report saved.\n")
    except Exception as exc:
        writer.set_error(exc, stage="model generation and simulation")
        raise
    verification = verify_simulation(spec, simulation_result)
    simulation_result["verification"] = verification.model_dump(mode="json")
    writer.set_result(simulation_result, simulation_result["verification"])
    print(json.dumps(simulation_result, indent=2))


if __name__ == "__main__":
    main()
