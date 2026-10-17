"""Local FastAPI interface for bounded baseline CircuitPilot runs."""

from __future__ import annotations

import json
import hashlib
import os
import re
import sqlite3
import shutil
import subprocess
import ipaddress
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

from .ir import ModelSpec
from .generator import generate_and_simulate
from .planner import get_provider_config
from .run_reports import RunReportWriter, new_run_id, write_repair_markdown
from .service import make_planner, run_baseline, run_repair_baseline


class StartRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=8000)
    provider: Literal["openai", "grok", "gemini"] | None = None
    model: str | None = Field(default=None, min_length=1, max_length=160)
    simulation_duration_seconds: float = Field(gt=0, le=86400)
    auto_repair: StrictBool = False

    @staticmethod
    def _check_finite_duration(value: float) -> float:
        import math
        if not math.isfinite(value):
            raise ValueError("simulation_duration_seconds must be finite")
        return value

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value.strip()

    @field_validator("simulation_duration_seconds")
    @classmethod
    def duration_is_finite(cls, value: float) -> float:
        return StartRunRequest._check_finite_duration(value)


class RunManager:
    """One MATLAB worker and a small, explicitly bounded accepted-run queue."""

    TERMINAL = {"completed", "failed", "interrupted"}
    RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{12}Z-[0-9a-f]{8}$")

    def __init__(self, output_root: str | Path, *, capacity: int = 2, timeout_seconds: float = 1800,
                 planner_factory: Any = make_planner, generator: Any = None,
                 max_repair_attempts: int = 2):
        if capacity < 1:
            raise ValueError("capacity must be at least one")
        if not 1 <= timeout_seconds <= 86400:
            raise ValueError("timeout_seconds must be between 1 and 86400")
        if isinstance(max_repair_attempts, bool) or not isinstance(max_repair_attempts, int) or not 0 <= max_repair_attempts <= 5:
            raise ValueError("max_repair_attempts must be between 0 and 5")
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.output_root / "run_registry.sqlite3"
        self.capacity = capacity
        self.timeout_seconds = float(timeout_seconds)
        self.max_repair_attempts = max_repair_attempts
        self.planner_factory = planner_factory
        self.generator = generator
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="circuitpilot-run")
        self.lock = threading.RLock()
        self.runs: dict[str, dict[str, Any]] = {}
        self.worker_run_id: str | None = None
        self._initialize_registry()
        self._recover_runs()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.registry_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    @contextmanager
    def _db(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize_registry(self) -> None:
        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, status TEXT NOT NULL, stage TEXT NOT NULL,
                run_root TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                timeout_seconds REAL NOT NULL, idempotency_key TEXT,
                auto_repair INTEGER NOT NULL DEFAULT 0,
                max_repair_attempts INTEGER NOT NULL DEFAULT 2
            )""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(runs)")}
            if "idempotency_key" not in columns:
                db.execute("ALTER TABLE runs ADD COLUMN idempotency_key TEXT")
            if "auto_repair" not in columns:
                db.execute("ALTER TABLE runs ADD COLUMN auto_repair INTEGER NOT NULL DEFAULT 0")
            if "max_repair_attempts" not in columns:
                db.execute("ALTER TABLE runs ADD COLUMN max_repair_attempts INTEGER NOT NULL DEFAULT 2")
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS runs_idempotency_key_unique "
                       "ON runs(idempotency_key) WHERE idempotency_key IS NOT NULL")

    def _reserve(self, run_id: str, timeout: float, idempotency_key: str | None, auto_repair: bool,
                 max_repair_attempts: int) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            active = db.execute("SELECT COUNT(*) FROM runs WHERE status NOT IN ('completed','failed','interrupted')").fetchone()[0]
            # A timed-out Python worker may still be blocked inside MATLAB. Count that
            # occupied worker separately after its public run has become terminal.
            occupied = 1 if self.worker_run_id and db.execute(
                "SELECT status FROM runs WHERE run_id=?", (self.worker_run_id,)
            ).fetchone() and db.execute(
                "SELECT status FROM runs WHERE run_id=?", (self.worker_run_id,)
            ).fetchone()[0] in self.TERMINAL else 0
            if active + occupied >= self.capacity:
                db.rollback()
                return False
            db.execute("""INSERT INTO runs
                (run_id, status, stage, run_root, created_at, updated_at, timeout_seconds,
                 idempotency_key, auto_repair, max_repair_attempts)
                VALUES (?, 'queued', 'queued', ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, run_id, now, now, timeout, idempotency_key, int(auto_repair), max_repair_attempts))
            return True

    def _transition(self, state: dict[str, Any], status: str, stage: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.lock:
            with self._db() as db:
                db.execute("UPDATE runs SET status=?, stage=?, updated_at=? WHERE run_id=?",
                           (status, stage, now, state["run_id"]))
            state["status"], state["stage"] = status, stage

    def _rows(self) -> list[sqlite3.Row]:
        with self._db() as db:
            return list(db.execute("SELECT * FROM runs ORDER BY created_at"))

    def _recover_runs(self) -> None:
        for row in self._rows():
            run_id = row["run_id"]
            if not self.RUN_ID_PATTERN.fullmatch(run_id) or row["run_root"] != run_id:
                continue
            run_root = (self.output_root / run_id).resolve()
            if run_root.parent != self.output_root:
                continue
            report_dir = run_root / "report"
            try:
                writer = RunReportWriter.resume(report_dir)
            except (OSError, ValueError, json.JSONDecodeError):
                writer = None
            status, stage = row["status"], row["stage"]
            if status not in self.TERMINAL:
                # A completed report and durable simulation artifact are the only
                # evidence sufficient to recover completion across a restart.
                completed = False
                simulation_path = run_root / "simulation.json"
                if writer and writer.data.get("lifecycle_status") == "completed" and simulation_path.is_file():
                    try:
                        simulation = json.loads(simulation_path.read_text(encoding="utf-8"))
                        completed = bool(writer.data.get("verification")) and simulation.get("simulation_status") in {"succeeded", "failed", "error"}
                    except (OSError, json.JSONDecodeError, AttributeError):
                        completed = False
                if completed:
                    status, stage = "completed", "completed"
                else:
                    if writer is None:
                        writer = RunReportWriter.recover(report_dir, run_id=run_id,
                            input_path="Previously accepted API run (request content omitted)")
                    writer.set_error(KeyboardInterrupt(), stage=stage or "server restart", interrupted=True)
                    status, stage = "interrupted", "interrupted"
                    if row["auto_repair"]:
                        repair_path = run_root / "repairs" / run_id / "repair_report.json"
                        try:
                            repair_data = json.loads(repair_path.read_text(encoding="utf-8"))
                            if repair_data.get("status") == "running":
                                repair_data["status"] = "interrupted"
                                repair_data["summary"] = "The API process restarted before the repair workflow completed."
                                for attempt in repair_data.get("attempts", []):
                                    if attempt.get("status") in {"planning", "simulating"}:
                                        attempt["status"] = "interrupted"
                                        attempt["failure_reason"] = "The API process restarted before this attempt completed."
                                repair_path.write_text(json.dumps(repair_data, indent=2), encoding="utf-8")
                                write_repair_markdown(
                                    repair_path.parent, repair_data,
                                    input_path=repair_data.get("input_path"),
                                    timestamp_utc=repair_data.get("timestamp_utc"),
                                )
                        except (OSError, json.JSONDecodeError, AttributeError, TypeError):
                            pass
                with self._db() as db:
                    db.execute("UPDATE runs SET status=?, stage=?, updated_at=? WHERE run_id=?",
                               (status, stage, datetime.now(timezone.utc).isoformat(), run_id))
            if writer is None:
                writer = RunReportWriter.recover(report_dir, run_id=run_id,
                    input_path="Previously accepted API run (request content omitted)")
                writer.set_error(RuntimeError("Saved API run report is unavailable"), stage=stage or "recovery")
                status, stage = "failed", "failed"
                with self._db() as db:
                    db.execute("UPDATE runs SET status=?, stage=?, updated_at=? WHERE run_id=?",
                               (status, stage, datetime.now(timezone.utc).isoformat(), run_id))
            simulation = None
            try:
                simulation = json.loads((run_root / "simulation.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
            self.runs[run_id] = {"run_id": run_id, "status": status, "stage": stage,
                                 "run_root": run_root, "writer": writer, "simulation": simulation,
                                 "auto_repair": bool(row["auto_repair"]),
                                 "max_repair_attempts": int(row["max_repair_attempts"]),
                                 "guard": threading.RLock(), "deadline_event": threading.Event(),
                                 "worker_started": False, "timer": None, "future": None}

    def submit(self, request: StartRunRequest, idempotency_key: str | None = None) -> dict[str, Any]:
        key_digest = None
        if idempotency_key is not None:
            if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", idempotency_key):
                raise HTTPException(status_code=400, detail="Invalid Idempotency-Key.")
            key_digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
        with self.lock:
            if key_digest:
                with self._db() as db:
                    existing = db.execute(
                        "SELECT run_id, status, auto_repair, max_repair_attempts FROM runs WHERE idempotency_key=?", (key_digest,)
                    ).fetchone()
                if existing:
                    run_id = existing["run_id"]
                    state = self.runs.get(run_id)
                    status = state["status"] if state else existing["status"]
                    return {"run_id": run_id, "status": status,
                            "status_url": f"/api/runs/{run_id}", "report_url": f"/api/runs/{run_id}/report",
                            "auto_repair": bool(existing["auto_repair"]),
                            "max_repair_attempts": existing["max_repair_attempts"] if existing["auto_repair"] else 0}
            run_id = new_run_id()
            if not self._reserve(run_id, self.timeout_seconds, key_digest, request.auto_repair,
                                 self.max_repair_attempts if request.auto_repair else 0):
                raise HTTPException(status_code=429, detail="CircuitPilot run queue is full; retry after an active run finishes.")
            run_root = self.output_root / run_id
            report_dir = run_root / "report"
            try:
                writer = RunReportWriter(report_dir, run_id=run_id, input_path="Natural-language prompt (content omitted)")
                writer.update(lifecycle_status="queued", stage="queued", repair={
                    "enabled": request.auto_repair, "attempt_count": 0,
                    "max_attempts": self.max_repair_attempts if request.auto_repair else 0,
                    "outcome": "queued" if request.auto_repair else "disabled",
                })
            except Exception:
                with self._db() as db:
                    db.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
                raise
            state = {"run_id": run_id, "status": "queued", "stage": "queued", "run_root": run_root,
                     "writer": writer, "guard": threading.RLock(), "deadline_event": threading.Event(),
                     "auto_repair": request.auto_repair,
                     "max_repair_attempts": self.max_repair_attempts if request.auto_repair else 0,
                     "worker_started": False, "timer": None, "future": None, "created_monotonic": time.monotonic()}
            self.runs[run_id] = state
            state["timer"] = threading.Timer(self.timeout_seconds, self._timeout_run, args=(state,))
            state["timer"].daemon = True
            state["timer"].start()
            state["future"] = self.executor.submit(self._execute, state, request)
            return {"run_id": run_id, "status": "queued", "status_url": f"/api/runs/{run_id}",
                    "report_url": f"/api/runs/{run_id}/report", "auto_repair": request.auto_repair,
                    "max_repair_attempts": state["max_repair_attempts"]}

    def _timeout_run(self, state: dict[str, Any]) -> None:
        with state["guard"]:
            if state["status"] in self.TERMINAL:
                return
            state["deadline_event"].set()
            future = state.get("future")
            if not state["worker_started"] and future is not None:
                future.cancel()
            exc = TimeoutError("CircuitPilot API execution deadline exceeded")
            if state.get("auto_repair"):
                with state["writer"]._lock:
                    repair = dict(state["writer"].data.get("repair") or {})
                repair.update(outcome="timed_out", summary="The API deadline elapsed; a blocked worker may still be active.")
                state["writer"].update(repair=repair)
            state["writer"].set_error(exc, stage=state.get("stage", "queued"))
            self._transition(state, "failed", "timed_out")

    def _execute(self, state: dict[str, Any], request: StartRunRequest) -> None:
        writer: RunReportWriter = state["writer"]

        with state["guard"]:
            if state["deadline_event"].is_set():
                return
            state["worker_started"] = True
            self.worker_run_id = state["run_id"]
            writer.update(lifecycle_status="running", stage="planning")
            self._transition(state, "running", "planning")

        def on_stage(stage: str) -> None:
            with state["guard"]:
                if state["deadline_event"].is_set():
                    raise TimeoutError("CircuitPilot API execution deadline exceeded")
                if stage == "completed":
                    # Expose completion only after the durable sample artifact is written.
                    return
                elif stage == "failed":
                    self._transition(state, "failed", state["stage"])
                elif stage == "interrupted":
                    self._transition(state, "interrupted", "interrupted")
                else:
                    self._transition(state, "running", stage)

        try:
            provider = request.provider or os.getenv("CIRCUITPILOT_PROVIDER", "openai")
            # Fail provider selection before creating a client; credentials are read server-side only.
            get_provider_config(provider)
            writer.update(stage="planning")
            planner = self.planner_factory(provider, request.model)
            run_args = dict(
                output_dir=state["run_root"], writer=writer, prompt=request.prompt,
                planner=planner, duration=request.simulation_duration_seconds,
                generator=self.generator or generate_and_simulate,
                on_stage=on_stage,
                on_spec=lambda spec: self._persist_spec(state, spec),
                should_stop=state["deadline_event"].is_set,
            )
            if request.auto_repair:
                result = run_repair_baseline(
                    **run_args, run_id=state["run_id"],
                    max_repair_attempts=state["max_repair_attempts"],
                )
            else:
                result = run_baseline(**run_args)
            # Durable run inputs and actual samples are scoped to this generated run directory.
            spec = planner.last_spec if hasattr(planner, "last_spec") else None
            # run_baseline returns the effective spec indirectly in its report metadata below.
            if state.get("spec") is not None:
                spec = state["spec"]
            if spec is not None:
                (state["run_root"] / "model_spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
            with state["guard"]:
                if state["deadline_event"].is_set():
                    return
                (state["run_root"] / "simulation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                state["simulation"] = result
                self._transition(state, "completed", "completed")
        except KeyboardInterrupt as exc:
            with state["guard"]:
                if not state["deadline_event"].is_set():
                    if request.auto_repair:
                        repair = dict(writer.data.get("repair") or {})
                        repair.update(outcome="interrupted", summary="The run was interrupted before the repair workflow completed.")
                        writer.update(repair=repair)
                    writer.set_error(exc, stage=state.get("stage", "unknown"), interrupted=True)
                    self._transition(state, "interrupted", "interrupted")
        except Exception as exc:
            # RunReportWriter records only safe exception categories and never raw messages.
            with state["guard"]:
                if not state["deadline_event"].is_set():
                    if request.auto_repair and (writer.data.get("repair") or {}).get("outcome") in {"queued", "running"}:
                        repair = dict(writer.data.get("repair") or {})
                        repair.update(outcome="error", summary=f"Repair workflow stopped during {state.get('stage', 'an unknown stage')}.")
                        writer.update(repair=repair)
                    if writer.data.get("lifecycle_status") not in {"error", "interrupted"}:
                        writer.set_error(exc, stage=state.get("stage", "unknown"))
                    self._transition(state, "failed", state.get("stage", "unknown"))
        finally:
            with self.lock:
                if self.worker_run_id == state["run_id"]:
                    self.worker_run_id = None
            timer = state.get("timer")
            if timer:
                timer.cancel()

    def get(self, run_id: str) -> dict[str, Any]:
        self._validate_run_id(run_id)
        with self.lock:
            state = self.runs.get(run_id)
            if state is None:
                raise HTTPException(status_code=404, detail="Unknown run ID.")
            with state["writer"]._lock:
                model_path = state["writer"].data.get("model_path")
            return {"run_id": run_id, "status": state["status"], "stage": state["stage"],
                    "model_path": model_path, "report_url": f"/api/runs/{run_id}/report",
                    "auto_repair": state.get("auto_repair", False),
                    "max_repair_attempts": state.get("max_repair_attempts", 0)}

    @staticmethod
    def _persist_spec(state: dict[str, Any], spec: ModelSpec) -> None:
        state["spec"] = spec
        (state["run_root"] / "model_spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")

    def report(self, run_id: str) -> dict[str, Any]:
        self._validate_run_id(run_id)
        with self.lock:
            state = self.runs.get(run_id)
            if state is None:
                raise HTTPException(status_code=404, detail="Unknown run ID.")
            with state["writer"]._lock:
                report = dict(state["writer"].data)
            run_root = state["run_root"]
        simulation = state.get("simulation")
        if simulation is None:
            path = run_root / "simulation.json"
            if path.is_file():
                try:
                    simulation = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    simulation = None
        spec_summary = None
        spec_path = run_root / "model_spec.json"
        if spec_path.is_file():
            try:
                spec = ModelSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
                spec_summary = {"name": spec.name, "components": [{"id": c.id, "type": c.type} for c in spec.components], "connections": [c.model_dump(mode="json") for c in spec.connections]}
            except Exception:
                spec_summary = None
        repair_path = run_root / "repairs" / run_id / "repair_report.json"
        repair = None
        if repair_path.is_file():
            try:
                repair = json.loads(repair_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                repair = {"status": "unavailable", "summary": "Repair history could not be read."}
        if (repair or {}).get("status") == "running" and report.get("error", {}).get("category") == "TIMEOUT":
            repair = dict(repair)
            repair["status"] = "timed_out"
            repair["summary"] = "The API deadline elapsed; the underlying planner or MATLAB worker may still be active."
        return {
            "report": report,
            "simulation": simulation,
            "model": spec_summary,
            "report_files": {
                "json": str((state["writer"].directory / "verification_report.json").relative_to(self.output_root)),
                "markdown": str((state["writer"].directory / "verification_report.md").relative_to(self.output_root)),
            },
            "repair": repair,
        }

    def markdown_report_path(self, run_id: str) -> Path:
        self._validate_run_id(run_id)
        with self.lock:
            state = self.runs.get(run_id)
            if state is None:
                raise HTTPException(status_code=404, detail="Unknown run ID.")
            path = state["writer"].directory / "verification_report.md"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Markdown report is not available yet.")
        return path

    def model_artifact_path(self, run_id: str, attempt_number: int | None = None) -> Path:
        """Resolve a saved model only from this run's trusted reports and artifacts."""
        self._validate_run_id(run_id)
        with self.lock:
            state = self.runs.get(run_id)
            if state is None:
                raise HTTPException(status_code=404, detail="Unknown run ID.")
            run_root = Path(state["run_root"]).resolve()
            if not run_root.is_relative_to(self.output_root) or run_root == self.output_root:
                raise HTTPException(status_code=404, detail="Model artifact is unavailable.")
            with state["writer"]._lock:
                report = dict(state["writer"].data)

        candidates: list[tuple[str, str | None]] = []
        repair_path = run_root / "repairs" / run_id / "repair_report.json"
        repair = None
        try:
            repair = json.loads(repair_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            repair = None
        attempts = repair.get("attempts", []) if isinstance(repair, dict) else []
        if attempt_number is not None:
            if isinstance(attempt_number, bool) or attempt_number < 0:
                raise HTTPException(status_code=404, detail="Model artifact is unavailable.")
            selected_attempt = next((item for item in attempts
                                     if isinstance(item, dict) and item.get("number") == attempt_number), None)
            if selected_attempt is None:
                raise HTTPException(status_code=404, detail="Model artifact is unavailable.")
            candidates.append((selected_attempt.get("model_path", ""), selected_attempt.get("attempt_dir")))
        else:
            best_number = repair.get("best_attempt") if isinstance(repair, dict) else None
            selected_attempt = next((item for item in attempts
                                     if isinstance(item, dict) and item.get("number") == best_number), None)
            if selected_attempt:
                candidates.append((selected_attempt.get("model_path", ""), selected_attempt.get("attempt_dir")))
            candidates.extend([
                (report.get("model_path", ""), None),
                ((state.get("simulation") or {}).get("model_path", ""), None),
            ])
            # Failed/interrupted repairs may not have a best_attempt, but a prior
            # attempt can still have a useful saved model. Prefer the latest one.
            if not selected_attempt:
                candidates.extend((item.get("model_path", ""), item.get("attempt_dir"))
                                  for item in reversed(attempts) if isinstance(item, dict))

        for raw_path, attempt_dir in candidates:
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            model_path = Path(raw_path)
            if not model_path.is_absolute():
                base = Path(attempt_dir) if isinstance(attempt_dir, str) and attempt_dir else run_root
                try:
                    base = base.resolve(strict=True)
                    if not base.is_relative_to(run_root):
                        continue
                except OSError:
                    continue
                model_path = base / model_path
            try:
                resolved = model_path.resolve(strict=True)
                resolved.relative_to(run_root)
            except (OSError, ValueError):
                continue
            if resolved.suffix.lower() == ".slx" and resolved.is_file():
                return resolved
        raise HTTPException(status_code=404, detail="No saved Simulink model is available for this run.")

    @staticmethod
    def _matlab_executable() -> str:
        configured = os.getenv("CIRCUITPILOT_MATLAB_EXECUTABLE")
        executable = configured.strip().strip('"') if configured else shutil.which("matlab")
        if not executable:
            raise HTTPException(status_code=503, detail=(
                "MATLAB was not found. Set CIRCUITPILOT_MATLAB_EXECUTABLE to the local MATLAB executable "
                "or add matlab to PATH. This action requires the API and browser to run on the same Windows machine."
            ))
        path = Path(executable)
        if not path.is_file() or path.name.casefold() not in {"matlab.exe", "matlab"}:
            raise HTTPException(status_code=503, detail="Configured MATLAB executable is unavailable or is not MATLAB.")
        return str(path.resolve())

    def open_model_in_matlab(self, run_id: str) -> dict[str, Any]:
        model_path = self.model_artifact_path(run_id)
        if os.name != "nt":
            raise HTTPException(status_code=501, detail="Open in MATLAB is available only from the local Windows development API.")
        executable = self._matlab_executable()
        matlab_string = str(model_path).replace("'", "''")
        command = f"open('{matlab_string}')"
        try:
            subprocess.Popen(
                [executable, "-desktop", "-nosplash", "-r", command],
                cwd=str(model_path.parent), stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                close_fds=True,
            )
        except OSError:
            raise HTTPException(status_code=503, detail="MATLAB could not be launched. Check the local executable configuration.") from None
        return {"status": "launched", "model_filename": model_path.name,
                "message": "MATLAB launch requested for the selected model."}

    @classmethod
    def _validate_run_id(cls, run_id: str) -> None:
        if not cls.RUN_ID_PATTERN.fullmatch(run_id):
            raise HTTPException(status_code=404, detail="Unknown run ID.")


def create_app(manager: RunManager | None = None) -> FastAPI:
    run_manager = manager or RunManager(
        os.getenv("CIRCUITPILOT_API_OUTPUT", "artifacts/api_runs"),
        capacity=int(os.getenv("CIRCUITPILOT_API_QUEUE_CAPACITY", "2")),
        timeout_seconds=float(os.getenv("CIRCUITPILOT_API_EXECUTION_TIMEOUT_SECONDS", "1800")),
        max_repair_attempts=int(os.getenv("CIRCUITPILOT_API_MAX_REPAIR_ATTEMPTS", "2")),
    )
    app = FastAPI(title="CircuitPilot API", version="0.1.0")
    origins = [origin.strip() for origin in os.getenv("CIRCUITPILOT_CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").split(",") if origin.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Idempotency-Key"],
    )
    app.state.run_manager = run_manager

    @app.post("/api/runs", status_code=202)
    def start_run(request: StartRunRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")) -> dict[str, Any]:
        # Direct endpoint invocation is used by offline tests; FastAPI's Header
        # marker remains the Python default in that context rather than None.
        if not isinstance(idempotency_key, str):
            idempotency_key = None
        return run_manager.submit(request, idempotency_key=idempotency_key)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        return run_manager.get(run_id)

    @app.get("/api/runs/{run_id}/report")
    def get_report(run_id: str) -> dict[str, Any]:
        return run_manager.report(run_id)

    @app.get("/api/runs/{run_id}/report.md")
    def get_markdown_report(run_id: str) -> FileResponse:
        # The generated run ID must resolve through the in-memory registry;
        # never accept a client-supplied filesystem path.
        return FileResponse(run_manager.markdown_report_path(run_id), media_type="text/markdown", filename="verification_report.md")

    @app.get("/api/runs/{run_id}/model")
    def download_model(run_id: str, attempt: int | None = None) -> FileResponse:
        if attempt is not None and attempt < 0:
            raise HTTPException(status_code=404, detail="Model artifact is unavailable.")
        model_path = run_manager.model_artifact_path(run_id, attempt_number=attempt)
        return FileResponse(
            model_path,
            media_type="application/vnd.mathworks.matlab-simulink-model",
            filename=model_path.name,
        )

    @app.post("/api/runs/{run_id}/open-in-matlab")
    def open_in_matlab(run_id: str, request: Request) -> dict[str, Any]:
        # This is intentionally a same-machine development action, not a remote
        # service capability. Never expose process launch to non-loopback clients.
        client_host = request.client.host if request.client else ""
        try:
            local_client = ipaddress.ip_address(client_host).is_loopback
        except ValueError:
            local_client = False
        if not local_client:
            raise HTTPException(status_code=403, detail="Open in MATLAB is restricted to local clients.")
        return run_manager.open_model_in_matlab(run_id)

    return app


app = create_app()
