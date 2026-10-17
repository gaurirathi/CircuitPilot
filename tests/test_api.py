"""Offline API integration tests; all planner and MATLAB work is mocked.

Call the synchronous FastAPI endpoint functions directly: this avoids opening
local sockets and keeps the suite deterministic on locked-down Windows hosts.
"""
import tempfile
import os
import threading
import time
import unittest
import json
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import HTTPException
from starlette.requests import Request
from pydantic import ValidationError

from circuitpilot.api import RunManager, StartRunRequest, create_app
from circuitpilot.ir import ModelSpec
from circuitpilot.run_reports import RunReportWriter, new_run_id
from circuitpilot.repair import RepairProposal


class FakePlanner:
    def __init__(self, spec): self.spec = spec
    def plan(self, prompt): return self.spec


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = ModelSpec.model_validate_json((Path(__file__).parents[1] / "examples" / "closed_loop_pid.json").read_text(encoding="utf-8"))
        cls.result = {"model_path": "ClosedLoopPID.slx", "model_opened": True, "diagram_update": "succeeded", "simulation_status": "succeeded", "time": [0, 1, 2, 3], "output": [0, 0.8, 1.05, 1.0]}

    def make_api(self, temp, *, spec=None, generator=None, capacity=2, timeout_seconds=1800):
        manager = RunManager(Path(temp) / "runs", capacity=capacity,
            timeout_seconds=timeout_seconds,
            planner_factory=lambda provider, model: FakePlanner(spec or self.spec),
            generator=generator or (lambda spec, output: dict(self.result)))
        app = create_app(manager)
        endpoints = {route.path: route.endpoint for route in app.routes if hasattr(route, "endpoint")}
        endpoints["_manager"] = manager
        return manager, endpoints

    def start(self, endpoints, prompt="PID loop", duration=7):
        return endpoints["/api/runs"](StartRunRequest(prompt=prompt, simulation_duration_seconds=duration))

    def wait_terminal(self, endpoints, run_id):
        for _ in range(200):
            status = endpoints["/api/runs/{run_id}"](run_id)
            if status["status"] in RunManager.TERMINAL:
                state = endpoints["_manager"].runs[run_id]
                if state and state.get("future"):
                    state["future"].result(timeout=2)
                return status
            time.sleep(0.01)
        self.fail("run did not reach a terminal state")

    def make_saved_model_run(self, temp, *, create_models=True):
        manager = RunManager(Path(temp) / "runs")
        run_id = new_run_id()
        run_root = manager.output_root / run_id
        original_dir = run_root / "repairs" / run_id / "attempt_00_original"
        repaired_dir = run_root / "repairs" / run_id / "attempt_01_repair"
        original_dir.mkdir(parents=True)
        repaired_dir.mkdir(parents=True)
        original_model = original_dir / "pid_control_system.slx"
        repaired_model = repaired_dir / "pid_control_system.slx"
        if create_models:
            original_model.write_bytes(b"original-model")
            repaired_model.write_bytes(b"repaired-model")
        writer = RunReportWriter(run_root / "report", run_id=run_id, input_path="test fixture")
        writer.update(lifecycle_status="completed", stage="verification complete",
                      model_path=str(repaired_model), simulation_status="succeeded",
                      verification={"status": "pass"})
        repair_dir = run_root / "repairs" / run_id
        repair_dir.mkdir(parents=True, exist_ok=True)
        (repair_dir / "repair_report.json").write_text(json.dumps({
            "status": "pass", "best_attempt": 1,
            "attempts": [
                {"number": 0, "kind": "original", "attempt_dir": str(original_dir), "model_path": str(original_model)},
                {"number": 1, "kind": "repair", "attempt_dir": str(repaired_dir), "model_path": str(repaired_model)},
            ],
        }), encoding="utf-8")
        state = {"run_id": run_id, "status": "completed", "stage": "completed",
                 "run_root": run_root, "writer": writer,
                 "simulation": {"model_path": str(repaired_model), "simulation_status": "succeeded"},
                 "auto_repair": True, "max_repair_attempts": 2}
        manager.runs[run_id] = state
        endpoints = {route.path: route.endpoint for route in create_app(manager).routes if hasattr(route, "endpoint")}
        return manager, endpoints, run_id, original_model, repaired_model

    def test_invalid_requests_reject_blanks_duration_and_credentials(self):
        for payload in [
            {"prompt": " ", "simulation_duration_seconds": 1},
            {"prompt": "valid", "simulation_duration_seconds": -1},
            {"prompt": "valid", "simulation_duration_seconds": 1, "api_key": "secret"},
            {"prompt": "valid", "simulation_duration_seconds": 1, "auto_repair": "true"},
            {"prompt": "valid", "simulation_duration_seconds": 1, "max_repair_attempts": 99},
        ]:
            with self.assertRaises(ValidationError): StartRunRequest.model_validate(payload)

    def test_cors_preflight_allows_frontend_origin_and_idempotency_header(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
            "CIRCUITPILOT_CORS_ORIGINS": "http://127.0.0.1:5173,http://localhost:5173",
        }, clear=False):
            manager = RunManager(Path(temp) / "runs", planner_factory=lambda provider, model: None)
            app = create_app(manager)
            messages = []
            scope = {
                "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1", "method": "OPTIONS", "scheme": "http",
                "path": "/api/runs", "raw_path": b"/api/runs", "query_string": b"",
                "root_path": "", "client": ("127.0.0.1", 12345), "server": ("127.0.0.1", 8000),
                "headers": [
                    (b"origin", b"http://127.0.0.1:5173"),
                    (b"access-control-request-method", b"POST"),
                    (b"access-control-request-headers", b"content-type,idempotency-key"),
                ],
            }

            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message):
                messages.append(message)

            # This preflight is answered entirely by CORSMiddleware and does not
            # need an event loop; drive the completed coroutine directly so the
            # test remains compatible with locked-down Windows event-loop setups.
            request = app(scope, receive, send)
            try:
                request.send(None)
            except StopIteration:
                pass
            else:
                self.fail("CORS preflight unexpectedly suspended")
            response = messages[0]
            headers = {key.decode().lower(): value.decode() for key, value in response["headers"]}
            self.assertEqual(response["status"], 200)
            self.assertEqual(headers.get("access-control-allow-origin"), "http://127.0.0.1:5173")
            allowed_headers = {header.strip().lower() for header in headers.get("access-control-allow-headers", "").split(",")}
            self.assertIn("content-type", allowed_headers)
            self.assertIn("idempotency-key", allowed_headers)
            self.assertIn("POST", headers.get("access-control-allow-methods", ""))

    def test_repair_attempt_configuration_is_defaulted_and_capped_server_side(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(os.environ, {"CIRCUITPILOT_API_OUTPUT": str(Path(temp) / "default-runs")}, clear=True):
                app = create_app()
                self.assertEqual(app.state.run_manager.max_repair_attempts, 2)
            for value in (-1, 6, 1.5, True):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    RunManager(Path(temp) / f"invalid-{value}", max_repair_attempts=value)
            self.assertEqual(RunManager(Path(temp) / "zero", max_repair_attempts=0).max_repair_attempts, 0)
            self.assertEqual(RunManager(Path(temp) / "five", max_repair_attempts=5).max_repair_attempts, 5)

    def test_unknown_run_ids_return_404(self):
        with tempfile.TemporaryDirectory() as temp:
            manager, endpoints = self.make_api(temp)
            with self.assertRaises(HTTPException) as status: endpoints["/api/runs/{run_id}"]("missing")
            self.assertEqual(status.exception.status_code, 404)
            with self.assertRaises(HTTPException) as report: endpoints["/api/runs/{run_id}/report"]("missing")
            self.assertEqual(report.exception.status_code, 404)
            with self.assertRaises(HTTPException) as markdown: endpoints["/api/runs/{run_id}/report.md"]("missing")
            self.assertEqual(markdown.exception.status_code, 404)
            with self.assertRaises(HTTPException) as traversal:
                endpoints["/api/runs/{run_id}/report.md"]("..\\verification_report.json")
            self.assertEqual(traversal.exception.status_code, 404)

    def test_model_download_selects_repaired_attempt_and_can_select_original_history(self):
        with tempfile.TemporaryDirectory() as temp:
            _, endpoints, run_id, original_model, repaired_model = self.make_saved_model_run(temp)
            selected = endpoints["/api/runs/{run_id}/model"](run_id)
            self.assertEqual(Path(selected.path), repaired_model.resolve())
            self.assertEqual(selected.filename, "pid_control_system.slx")
            self.assertEqual(selected.media_type, "application/vnd.mathworks.matlab-simulink-model")
            original = endpoints["/api/runs/{run_id}/model"](run_id, attempt=0)
            self.assertEqual(Path(original.path), original_model.resolve())

    def test_missing_or_unsafe_model_artifacts_return_404(self):
        with tempfile.TemporaryDirectory() as temp:
            manager, endpoints, run_id, _, _ = self.make_saved_model_run(temp, create_models=False)
            with self.assertRaises(HTTPException) as missing:
                endpoints["/api/runs/{run_id}/model"](run_id)
            self.assertEqual(missing.exception.status_code, 404)
            with self.assertRaises(HTTPException) as invalid_id:
                endpoints["/api/runs/{run_id}/model"]("..\\secret")
            self.assertEqual(invalid_id.exception.status_code, 404)

            outside = Path(temp) / "outside.slx"
            outside.write_bytes(b"not-for-download")
            repair_file = manager.runs[run_id]["run_root"] / "repairs" / run_id / "repair_report.json"
            repair = json.loads(repair_file.read_text(encoding="utf-8"))
            repair["attempts"][1]["model_path"] = str(outside)
            repair_file.write_text(json.dumps(repair), encoding="utf-8")
            manager.runs[run_id]["writer"].update(model_path=str(outside))
            manager.runs[run_id]["simulation"]["model_path"] = str(outside)
            with self.assertRaises(HTTPException) as traversal:
                endpoints["/api/runs/{run_id}/model"](run_id)
            self.assertEqual(traversal.exception.status_code, 404)

    def test_open_in_matlab_launches_selected_artifact_with_mocked_process(self):
        with tempfile.TemporaryDirectory() as temp:
            _, endpoints, run_id, _, selected_model = self.make_saved_model_run(temp)
            executable = Path(temp) / "matlab.exe"
            executable.write_bytes(b"mock executable")
            request = Request({"type": "http", "method": "POST", "path": f"/api/runs/{run_id}/open-in-matlab",
                "headers": [], "query_string": b"", "client": ("127.0.0.1", 1234), "server": ("127.0.0.1", 8000)})
            with patch("circuitpilot.api.os.name", "nt"), \
                 patch.dict(os.environ, {"CIRCUITPILOT_MATLAB_EXECUTABLE": str(executable)}), \
                 patch("circuitpilot.api.subprocess.Popen") as launch:
                result = endpoints["/api/runs/{run_id}/open-in-matlab"](run_id, request)
            self.assertEqual(result["status"], "launched")
            self.assertEqual(result["model_filename"], "pid_control_system.slx")
            args, kwargs = launch.call_args
            self.assertEqual(args[0][0], str(executable.resolve()))
            self.assertIn(f"open('{selected_model.resolve()}')", args[0][-1])
            self.assertEqual(kwargs["cwd"], str(selected_model.parent))
            self.assertNotIn("shell", kwargs)

    def test_matlab_unavailable_launch_failure_and_remote_client_are_safe(self):
        with tempfile.TemporaryDirectory() as temp:
            _, endpoints, run_id, _, _ = self.make_saved_model_run(temp)
            request = Request({"type": "http", "method": "POST", "path": f"/api/runs/{run_id}/open-in-matlab",
                "headers": [], "query_string": b"", "client": ("127.0.0.1", 1234), "server": ("127.0.0.1", 8000)})
            with patch("circuitpilot.api.os.name", "nt"), \
                 patch.dict(os.environ, {}, clear=False), \
                 patch.dict(os.environ, {"CIRCUITPILOT_MATLAB_EXECUTABLE": str(Path(temp) / "missing.exe")}, clear=False):
                with self.assertRaises(HTTPException) as unavailable:
                    endpoints["/api/runs/{run_id}/open-in-matlab"](run_id, request)
                self.assertEqual(unavailable.exception.status_code, 503)
            missing_config = Request({**request.scope, "client": ("192.0.2.5", 1234)})
            with self.assertRaises(HTTPException) as remote:
                endpoints["/api/runs/{run_id}/open-in-matlab"](run_id, missing_config)
            self.assertEqual(remote.exception.status_code, 403)

            executable = Path(temp) / "matlab.exe"
            executable.write_bytes(b"mock executable")
            with patch("circuitpilot.api.os.name", "nt"), \
                 patch.dict(os.environ, {"CIRCUITPILOT_MATLAB_EXECUTABLE": str(executable)}), \
                 patch("circuitpilot.api.subprocess.Popen", side_effect=OSError("hidden launch detail")):
                with self.assertRaises(HTTPException) as failed:
                    endpoints["/api/runs/{run_id}/open-in-matlab"](run_id, request)
            self.assertEqual(failed.exception.status_code, 503)
            self.assertNotIn("hidden launch detail", str(failed.exception.detail))

    def test_status_polling_and_final_report_include_real_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            manager, endpoints = self.make_api(temp)
            started = self.start(endpoints)
            self.assertEqual(started["status"], "queued")
            status = self.wait_terminal(endpoints, started["run_id"])
            self.assertEqual(status["status"], "completed")
            report = endpoints["/api/runs/{run_id}/report"](started["run_id"])
            self.assertEqual(report["report"]["simulation_status"], "succeeded")
            self.assertEqual(report["simulation"]["time"], self.result["time"])
            self.assertEqual(report["model"]["name"], self.spec.name)
            self.assertTrue((Path(temp) / "runs" / started["run_id"] / "model_spec.json").exists())
            markdown = endpoints["/api/runs/{run_id}/report.md"](started["run_id"])
            self.assertTrue(Path(markdown.path).is_file())
            self.assertEqual(markdown.media_type, "text/markdown")
            with manager._db() as db:
                row = db.execute("SELECT status, stage FROM runs WHERE run_id=?", (started["run_id"],)).fetchone()
                self.assertEqual((row["status"], row["stage"]), ("completed", "completed"))
            self.assertFalse(status["auto_repair"])
            self.assertFalse(report["report"]["repair"]["enabled"])

    def test_opt_in_repair_uses_isolated_fake_attempts_and_persists_history(self):
        data = self.spec.model_dump(mode="json")
        data["verification"] = {"max_tracking_error": 0.01, "max_overshoot_percent": 5, "max_settling_time": 5}
        repair_spec = ModelSpec.model_validate(data)
        class RepairFakePlanner(FakePlanner):
            def __init__(self, spec):
                super().__init__(spec)
                self.repair_calls = 0
            def propose_repair(self, spec, diagnosis):
                self.repair_calls += 1
                return RepairProposal(reason="Measured overshoot was high", changes_summary="Adjust existing PID P gain", parameter_updates={"controller": {"P": 3}})
        planner = RepairFakePlanner(repair_spec)
        outputs = iter([
            {"model_path": "original.slx", "simulation_status": "succeeded", "time": [0, 1, 2, 3, 10], "output": [0, 0.8, 1.2, 1.05, 1.0]},
            {"model_path": "repaired.slx", "simulation_status": "succeeded", "time": [0, 1, 2, 3, 10], "output": [0, 0.8, 1.0, 1.0, 1.0]},
        ])
        generated = []
        def fake_generator(spec, output_dir):
            generated.append(Path(output_dir))
            return next(outputs)
        with tempfile.TemporaryDirectory() as temp:
            manager = RunManager(Path(temp) / "runs", planner_factory=lambda provider, model: planner, generator=fake_generator)
            endpoints = {route.path: route.endpoint for route in create_app(manager).routes if hasattr(route, "endpoint")}
            endpoints["_manager"] = manager
            started = endpoints["/api/runs"](StartRunRequest(prompt="PID loop", simulation_duration_seconds=10, auto_repair=True))
            self.assertTrue(started["auto_repair"])
            self.assertEqual(self.wait_terminal(endpoints, started["run_id"])["status"], "completed")
            payload = endpoints["/api/runs/{run_id}/report"](started["run_id"])
            history = payload["repair"]
            self.assertEqual(history["status"], "pass")
            self.assertEqual([item["status"] for item in history["attempts"]], ["fail", "pass"])
            self.assertEqual(len(set(generated)), 2)
            self.assertTrue(all("attempt_" in str(path) for path in generated))
            self.assertEqual(planner.repair_calls, 1)
            self.assertEqual(payload["report"]["repair"]["outcome"], "pass")
            self.assertEqual(json.loads((Path(history["attempts"][0]["attempt_dir"]) / "spec.json").read_text())["components"][2]["parameters"]["P"], 2)

    def test_registry_restart_marks_active_run_interrupted(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "runs"
            manager = RunManager(output)
            run_id = new_run_id()
            run_root = output / run_id
            writer = RunReportWriter(run_root / "report", run_id=run_id, input_path="Natural-language prompt (content omitted)")
            writer.update(lifecycle_status="running", stage="simulation")
            with manager._db() as db:
                db.execute("INSERT INTO runs (run_id, status, stage, run_root, created_at, updated_at, timeout_seconds) VALUES (?, 'running', 'simulation', ?, ?, ?, ?)",
                    (run_id, run_id, "2026-10-17T00:00:00+00:00", "2026-10-17T00:00:00+00:00", 1800))
            recovered = RunManager(output)
            status = recovered.get(run_id)
            self.assertEqual((status["status"], status["stage"]), ("interrupted", "interrupted"))
            report = recovered.report(run_id)["report"]
            self.assertEqual(report["error"]["category"], "INTERRUPTED")
            saved = json.loads((run_root / "report" / "verification_report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["lifecycle_status"], "interrupted")

    def test_restart_preserves_opt_in_repair_choice_and_marks_active_run_interrupted(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "runs"
            manager = RunManager(output)
            run_id = new_run_id()
            run_root = output / run_id
            writer = RunReportWriter(run_root / "report", run_id=run_id, input_path="Natural-language prompt (content omitted)")
            writer.update(lifecycle_status="running", stage="simulation", repair={"enabled": True, "attempt_count": 0, "max_attempts": 2, "outcome": "running"})
            repair_dir = run_root / "repairs" / run_id
            repair_dir.mkdir(parents=True)
            (repair_dir / "repair_report.json").write_text(json.dumps({
                "status": "running", "run_dir": str(repair_dir), "max_repair_attempts": 2,
                "best_attempt": None, "attempts": [{"number": 0, "kind": "original", "status": "simulating",
                    "attempt_dir": str(repair_dir / "attempt_00_original"), "changes_summary": "Original user specification."}],
                "summary": "Simulation in progress.", "run_id": run_id, "timestamp_utc": "2026-10-17T00:00:00+00:00",
            }), encoding="utf-8")
            with manager._db() as db:
                db.execute("INSERT INTO runs (run_id, status, stage, run_root, created_at, updated_at, timeout_seconds, auto_repair) VALUES (?, 'running', 'simulation', ?, ?, ?, ?, 1)",
                    (run_id, run_id, "2026-10-17T00:00:00+00:00", "2026-10-17T00:00:00+00:00", 1800))
            recovered = RunManager(output)
            status = recovered.get(run_id)
            self.assertTrue(status["auto_repair"])
            self.assertEqual(status["status"], "interrupted")
            self.assertTrue(recovered.report(run_id)["report"]["repair"]["enabled"])
            self.assertEqual(recovered.report(run_id)["repair"]["status"], "interrupted")

    def test_registry_migrates_existing_sqlite_schema_for_idempotency(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "runs"
            output.mkdir()
            registry = output / "run_registry.sqlite3"
            with sqlite3.connect(registry) as db:
                db.execute("""CREATE TABLE runs (
                    run_id TEXT PRIMARY KEY, status TEXT NOT NULL, stage TEXT NOT NULL,
                    run_root TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    timeout_seconds REAL NOT NULL
                )""")
            manager = RunManager(output)
            with manager._db() as db:
                columns = {row[1] for row in db.execute("PRAGMA table_info(runs)")}
                self.assertIn("idempotency_key", columns)
                indexes = {row[1] for row in db.execute("PRAGMA index_list(runs)")}
                self.assertIn("runs_idempotency_key_unique", indexes)

    def test_restart_keeps_run_completed_when_report_and_simulation_are_durable(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "runs"
            manager = RunManager(output)
            run_id = new_run_id()
            run_root = output / run_id
            writer = RunReportWriter(run_root / "report", run_id=run_id, input_path="Natural-language prompt (content omitted)")
            writer.update(lifecycle_status="completed", stage="verification complete", verification={"status": "inconclusive"}, simulation_status="succeeded")
            (run_root / "simulation.json").write_text(json.dumps({"simulation_status": "succeeded", "verification": {"status": "inconclusive"}}), encoding="utf-8")
            with manager._db() as db:
                db.execute("INSERT INTO runs (run_id, status, stage, run_root, created_at, updated_at, timeout_seconds) VALUES (?, 'running', 'verification', ?, ?, ?, ?)",
                    (run_id, run_id, "2026-10-17T00:00:00+00:00", "2026-10-17T00:00:00+00:00", 1800))
            recovered = RunManager(output)
            status = recovered.get(run_id)
            self.assertEqual((status["status"], status["stage"]), ("completed", "completed"))

    def test_execution_timeout_persists_terminal_report_safely(self):
        entered, release = threading.Event(), threading.Event()
        secret = "Authorization: Bearer do-not-persist"
        def blocked(spec, output):
            entered.set()
            release.wait(3)
            raise RuntimeError(secret)
        with tempfile.TemporaryDirectory() as temp:
            manager, endpoints = self.make_api(temp, generator=blocked, timeout_seconds=1)
            try:
                started = self.start(endpoints, "timeout", 1)
                self.assertTrue(entered.wait(2))
                status = None
                for _ in range(200):
                    status = endpoints["/api/runs/{run_id}"](started["run_id"])
                    if status["status"] in RunManager.TERMINAL:
                        break
                    time.sleep(0.01)
                self.assertEqual(status["status"], "failed")
                report = endpoints["/api/runs/{run_id}/report"](started["run_id"])
                self.assertEqual(report["report"]["error"]["category"], "TIMEOUT")
                self.assertNotIn("do-not-persist", str(report))
                with manager._db() as db:
                    row = db.execute("SELECT status FROM runs WHERE run_id=?", (started["run_id"],)).fetchone()
                    self.assertEqual(row["status"], "failed")
            finally:
                release.set()
            endpoints["_manager"].runs[started["run_id"]]["future"].result(timeout=2)
            final_report = endpoints["/api/runs/{run_id}/report"](started["run_id"])
            self.assertEqual(final_report["report"]["error"]["category"], "TIMEOUT")
            self.assertIsNone(final_report["simulation"])
            for _ in range(200):
                if manager.worker_run_id is None:
                    break
                time.sleep(0.01)

    def test_timeout_configuration_rejects_out_of_range_values(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                RunManager(Path(temp) / "runs", timeout_seconds=0)

    def test_capacity_rejects_excess_work(self):
        entered, release = threading.Event(), threading.Event()
        def blocking(spec, output):
            if not entered.is_set():
                entered.set()
                release.wait(2)
            return dict(self.result)
        with tempfile.TemporaryDirectory() as temp:
            manager, endpoints = self.make_api(temp, generator=blocking, capacity=2)
            try:
                first = self.start(endpoints, "first", 1)
                self.assertTrue(entered.wait(2))
                second = self.start(endpoints, "second", 1)
                with self.assertRaises(HTTPException) as full: self.start(endpoints, "second", 1)
                self.assertEqual(full.exception.status_code, 429)
            finally:
                release.set()
            self.wait_terminal(endpoints, first["run_id"])
            self.wait_terminal(endpoints, second["run_id"])

    def test_idempotency_key_deduplicates_api_submission_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            manager, endpoints = self.make_api(temp)
            request = StartRunRequest(prompt="same submission", simulation_duration_seconds=3)
            first = endpoints["/api/runs"](request, "frontend-request-123")
            self.wait_terminal(endpoints, first["run_id"])
            duplicate = endpoints["/api/runs"](request, "frontend-request-123")
            self.assertEqual(duplicate["run_id"], first["run_id"])
            with manager._db() as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 1)

            recovered = RunManager(Path(temp) / "runs")
            recovered_endpoints = {route.path: route.endpoint for route in create_app(recovered).routes if hasattr(route, "endpoint")}
            after_restart = recovered_endpoints["/api/runs"](request, "frontend-request-123")
            self.assertEqual(after_restart["run_id"], first["run_id"])
            self.assertEqual(after_restart["status"], "completed")

    def test_idempotency_reuses_the_original_repair_opt_in(self):
        with tempfile.TemporaryDirectory() as temp:
            manager, endpoints = self.make_api(temp)
            accepted_request = StartRunRequest(prompt="same", simulation_duration_seconds=2, auto_repair=True)
            accepted = endpoints["/api/runs"](accepted_request, "repair-idempotency")
            self.wait_terminal(endpoints, accepted["run_id"])
            duplicate = endpoints["/api/runs"](StartRunRequest(prompt="different", simulation_duration_seconds=8, auto_repair=False), "repair-idempotency")
            self.assertEqual(duplicate["run_id"], accepted["run_id"])
            self.assertTrue(duplicate["auto_repair"])

    def test_invalid_idempotency_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            manager, endpoints = self.make_api(temp)
            request = StartRunRequest(prompt="PID loop", simulation_duration_seconds=1)
            with self.assertRaises(HTTPException) as invalid:
                endpoints["/api/runs"](request, "../bad")
            self.assertEqual(invalid.exception.status_code, 400)

    def test_invalid_graph_is_rejected_before_generator(self):
        data = self.spec.model_dump(mode="json")
        data["connections"][0]["target"] = "missing-component"
        invalid = ModelSpec.model_validate(data)
        generator = Mock(return_value=dict(self.result))
        with tempfile.TemporaryDirectory() as temp:
            manager, endpoints = self.make_api(temp, spec=invalid, generator=generator)
            started = self.start(endpoints, "invalid", 1)
            self.assertEqual(self.wait_terminal(endpoints, started["run_id"])["status"], "failed")
            generator.assert_not_called()
            report = endpoints["/api/runs/{run_id}/report"](started["run_id"])["report"]
            self.assertEqual(report["error"]["category"], "VALIDATION")

    def test_sensitive_exception_details_are_not_returned(self):
        secret = "api_key=super-secret Authorization: Bearer hidden"
        def fail(spec, output): raise RuntimeError(secret)
        with tempfile.TemporaryDirectory() as temp:
            manager, endpoints = self.make_api(temp, generator=fail)
            started = self.start(endpoints, "fails", 1)
            self.wait_terminal(endpoints, started["run_id"])
            report = endpoints["/api/runs/{run_id}/report"](started["run_id"])
            text = str(report)
            self.assertEqual(report["report"]["error"]["exception_type"], "RuntimeError")
            self.assertNotIn("super-secret", text)
            self.assertNotIn("Bearer hidden", text)


if __name__ == "__main__": unittest.main()
