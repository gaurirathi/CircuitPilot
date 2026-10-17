import json
import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from circuitpilot.cli import main
from circuitpilot.ir import ModelSpec
from circuitpilot.planner import (
    ClarificationRequired,
    GeminiPlanner,
    GEMINI_INTERACTIONS_API_VERSION,
    MockPlanner,
    OpenAIPlanner,
    PlannerError,
    PlannerEnvelope,
    ProviderRequestError,
    SYSTEM_INSTRUCTIONS,
    create_provider_client,
    get_provider_config,
    load_project_dotenv,
)
from circuitpilot.repair import RepairDiagnosis, RepairProposal
from circuitpilot.run_reports import RunReportWriter


class MockPlannerTests(unittest.TestCase):
    def setUp(self):
        self.prompt = "Build a unit-step closed-loop PID with the specified plant."
        self.graph = json.loads((Path(__file__).parents[1] / "examples/closed_loop_pid.json").read_text())
        self.envelope = {
            "status": "ready",
            "graph_json": json.dumps(self.graph),
            "message": "",
        }

    def test_successful_plan_returns_validated_graph(self):
        spec = MockPlanner({self.prompt: self.envelope}).plan(self.prompt)
        self.assertEqual(spec.name, "ClosedLoopPID")
        self.assertEqual(len(spec.components), 5)

    def test_malformed_envelope_is_rejected(self):
        with self.assertRaisesRegex(PlannerError, "malformed structured output"):
            MockPlanner({self.prompt: "not json"}).plan(self.prompt)

    def test_unsupported_component_is_rejected(self):
        graph = json.loads(json.dumps(self.graph))
        graph["components"][0]["type"] = "MOSFET"
        envelope = {**self.envelope, "graph_json": json.dumps(graph)}
        with self.assertRaisesRegex(PlannerError, r"Unsupported component type 'MOSFET'.*Supported types:"):
            MockPlanner({self.prompt: envelope}).plan(self.prompt)

    def test_invalid_connections_are_rejected(self):
        graph = json.loads(json.dumps(self.graph))
        graph["connections"][0]["target"] = "unknown"
        envelope = {**self.envelope, "graph_json": json.dumps(graph)}
        with self.assertRaisesRegex(PlannerError, "unknown component"):
            MockPlanner({self.prompt: envelope}).plan(self.prompt)

    def test_missing_required_parameters_are_rejected(self):
        graph = json.loads(json.dumps(self.graph))
        graph["components"][3]["parameters"].pop("Denominator")
        envelope = {**self.envelope, "graph_json": json.dumps(graph)}
        with self.assertRaisesRegex(PlannerError, "Missing required Transfer Fcn parameter"):
            MockPlanner({self.prompt: envelope}).plan(self.prompt)

    def test_clarification_response_is_not_a_graph(self):
        response = {
            "status": "clarification",
            "graph_json": "",
            "message": "Please specify the plant transfer function.",
        }
        with self.assertRaisesRegex(ClarificationRequired, "specify the plant"):
            MockPlanner({self.prompt: response}).plan(self.prompt)


class PlannerCliTests(unittest.TestCase):
    def test_prompt_cli_saves_validated_graph_and_calls_generator(self):
        prompt = "Create the sample PID model"
        spec = ModelSpec.model_validate_json(
            (Path(__file__).parents[1] / "examples/closed_loop_pid.json").read_text()
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            graph_path = root / "planned.json"
            with patch("sys.argv", ["circuitpilot", "--prompt", prompt, "--graph-output", str(graph_path), "--output", str(root / "artifacts")]), \
                 patch.dict(os.environ, {"CIRCUITPILOT_PROVIDER": "openai"}), \
                 patch("circuitpilot.cli.OpenAIPlanner") as planner_cls, \
                 patch("circuitpilot.cli.generate_and_simulate", return_value={"simulation_status": "succeeded", "time": [0, 1], "output": [0, 1]}) as generate:
                planner_cls.return_value.plan.return_value = spec
                main()
            self.assertEqual(ModelSpec.model_validate_json(graph_path.read_text()).name, spec.name)
            generate.assert_called_once_with(spec, root / "artifacts")

    def test_prompt_cli_passes_explicit_provider_and_model(self):
        spec = ModelSpec.model_validate_json(
            (Path(__file__).parents[1] / "examples/closed_loop_pid.json").read_text()
        )
        with tempfile.TemporaryDirectory() as temp:
            with patch("sys.argv", ["circuitpilot", "--prompt", "make a model", "--provider", "grok", "--model", "grok-4.7", "--output", temp]), \
                 patch("circuitpilot.cli.OpenAIPlanner") as planner_cls, \
                 patch("circuitpilot.cli.generate_and_simulate", return_value={"simulation_status": "succeeded", "time": [0, 1], "output": [0, 1]}):
                planner_cls.return_value.plan.return_value = spec
                main()
            planner_cls.assert_called_once_with(provider="grok", model="grok-4.7")

    def test_prompt_cli_selects_gemini_planner(self):
        spec = ModelSpec.model_validate_json(
            (Path(__file__).parents[1] / "examples/closed_loop_pid.json").read_text()
        )
        with tempfile.TemporaryDirectory() as temp:
            with patch("sys.argv", ["circuitpilot", "--prompt", "make a model", "--provider", "gemini", "--model", "gemini-test", "--output", temp]), \
                 patch("circuitpilot.cli.GeminiPlanner") as planner_cls, \
                 patch("circuitpilot.cli.generate_and_simulate", return_value={"simulation_status": "succeeded", "time": [0, 1], "output": [0, 1]}):
                planner_cls.return_value.plan.return_value = spec
                main()
            planner_cls.assert_called_once_with(model="gemini-test")

    def test_prompt_cli_runs_gemini_diagnostic_before_planning(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch("sys.argv", ["circuitpilot", "--prompt", "make a model", "--provider", "gemini", "--diagnose-gemini", "--output", temp]), \
                 patch("circuitpilot.cli.GeminiPlanner") as planner_cls, \
                 patch("circuitpilot.cli.generate_and_simulate", return_value={"simulation_status": "succeeded", "time": [0, 1], "output": [0, 1]}):
                planner_cls.return_value.model = "gemini-diagnostic-test"
                planner_cls.return_value.credential_source = "GEMINI_API_KEY"
                planner_cls.return_value.credential_configured = True
                planner_cls.return_value.api_endpoint = "https://generativelanguage.googleapis.com/v1beta"
                planner_cls.return_value.diagnose_api.return_value = [
                    {"probe": name, "success": True, "detail": "ok"}
                    for name in (
                        "interactions_plain_text",
                        "interactions_structured_json",
                        "interactions_circuitpilot_schema",
                    )
                ]
                main()
            planner_cls.return_value.diagnose_api.assert_called_once_with()
            planner_cls.return_value.plan.assert_not_called()

    def test_ctrl_c_during_gemini_planning_writes_interrupted_report_and_exits_cleanly(self):
        class InterruptedPlanner:
            def plan(self, prompt):
                raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as temp:
            with patch("sys.argv", ["circuitpilot", "--prompt", "make a model", "--provider", "gemini", "--output", temp]), \
                 patch("circuitpilot.cli.GeminiPlanner", return_value=InterruptedPlanner()):
                with self.assertRaises(SystemExit) as raised:
                    main()
            self.assertEqual(raised.exception.code, 130)
            report_dir = next((Path(temp) / "run_reports").iterdir())
            report = json.loads((report_dir / "verification_report.json").read_text(encoding="utf-8"))
            markdown = (report_dir / "verification_report.md").read_text(encoding="utf-8")
        self.assertEqual(report["overall_status"], "error")
        self.assertEqual(report["lifecycle_status"], "interrupted")
        self.assertEqual(report["error"]["category"], "INTERRUPTED")
        self.assertIn("Overall status: **ERROR**", markdown)

    def test_rate_limit_error_category_is_checkpointed_without_request_details(self):
        secret = "private-key-value"
        error = ProviderRequestError(
            "rate-limited request " + secret,
            retry_count=2,
            transient=True,
            status_code=429,
            provider_error="HTTP 429 request body " + secret,
            failure_type="provider",
        )
        with tempfile.TemporaryDirectory() as temp:
            writer = RunReportWriter(Path(temp) / "run", run_id="rate-run", input_path="prompt omitted")
            writer.set_error(error, stage="planning")
            report_path = writer.directory / "verification_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            markdown = (writer.directory / "verification_report.md").read_text(encoding="utf-8")
        self.assertEqual(report["overall_status"], "error")
        self.assertEqual(report["error"]["category"], "RATE_LIMIT")
        self.assertEqual(report["error"]["http_status"], 429)
        self.assertIn("rate-limited", markdown.lower())
        self.assertNotIn(secret, markdown + json.dumps(report))


class StructuredAdapterTests(unittest.TestCase):
    @staticmethod
    def http_error(status, retry_after=None):
        error = RuntimeError(f"mock HTTP {status}")
        response = type("Response", (), {
            "status_code": status,
            "text": f"provider returned HTTP {status}",
            "headers": ({"Retry-After": str(retry_after)} if retry_after is not None else {}),
        })()
        error.status_code = status
        error.response = response
        return error

    def test_gemini_503_retries_and_respects_retry_after(self):
        proposal = RepairProposal(
            reason="Reduce overshoot", changes_summary="Change existing P gain.",
            parameter_updates={"controller": {"P": 3}},
        )
        diagnosis = RepairDiagnosis(summary="Overshoot failed", failed_checks=[{
            "name": "peak_overshoot", "value": 6.4, "unit": "%", "criterion": "<= 5%",
            "reason": "Overshoot exceeded its limit.",
        }])

        class Interactions:
            calls = 0

            def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise StructuredAdapterTests.http_error(503, retry_after=3)
                return type("Response", (), {"output_text": proposal.model_dump_json()})()

        client = type("Client", (), {"interactions": Interactions()})()
        with patch("circuitpilot.planner.time.sleep") as sleep, \
             patch("circuitpilot.planner.random.uniform", return_value=0.5):
            actual = GeminiPlanner(client=client, max_retries=2).propose_repair(
                ModelSpec.model_validate_json((Path(__file__).parents[1] / "examples/closed_loop_pid.json").read_text()),
                diagnosis,
            )
        self.assertEqual(actual, proposal)
        self.assertEqual(client.interactions.calls, 2)
        sleep.assert_called_once_with(3.0)

    def test_openai_429_exhaustion_records_transient_error_and_retry_count(self):
        diagnosis = RepairDiagnosis(summary="Failed", failed_checks=[{
            "name": "settling_time", "value": 6.1, "unit": "s", "criterion": "<= 5 s",
            "reason": "Slow response.",
        }])

        class Responses:
            calls = 0

            def parse(self, **kwargs):
                self.calls += 1
                raise StructuredAdapterTests.http_error(429, retry_after=0.25)

        client = type("Client", (), {"responses": Responses()})()
        spec = ModelSpec.model_validate_json(
            (Path(__file__).parents[1] / "examples/closed_loop_pid.json").read_text()
        )
        with patch("circuitpilot.planner.time.sleep") as sleep, \
             patch("circuitpilot.planner.random.uniform", return_value=0.5):
            with self.assertRaises(ProviderRequestError) as raised:
                OpenAIPlanner(provider="openai", client=client, max_retries=1).propose_repair(spec, diagnosis)
        error = raised.exception
        self.assertEqual(client.responses.calls, 2)
        self.assertEqual(error.status_code, 429)
        self.assertEqual(error.retry_count, 1)
        self.assertTrue(error.transient)
        self.assertIn("HTTP 429", error.provider_error)
        self.assertEqual(sleep.call_args_list[0].args, (0.5,))

    def test_permanent_4xx_is_not_retried(self):
        for status in (400, 401):
            with self.subTest(status=status):
                class Interactions:
                    calls = 0

                    def create(inner_self, **kwargs):
                        inner_self.calls += 1
                        raise StructuredAdapterTests.http_error(status)

                client = type("Client", (), {"interactions": Interactions()})()
                with patch("circuitpilot.planner.time.sleep") as sleep:
                    with self.assertRaises(ProviderRequestError) as raised:
                        GeminiPlanner(client=client, max_retries=2).plan("plan")
                self.assertEqual(client.interactions.calls, 1)
                self.assertEqual(raised.exception.retry_count, 0)
                self.assertFalse(raised.exception.transient)
                sleep.assert_not_called()

    def test_timeout_is_bounded_and_classified_separately(self):
        attempts = 0

        def request():
            nonlocal attempts
            attempts += 1
            raise TimeoutError("mock request timed out")

        with patch("circuitpilot.planner.time.sleep"), \
             patch("circuitpilot.planner.random.uniform", return_value=0.1):
            with self.assertRaises(ProviderRequestError) as raised:
                from circuitpilot.planner import _request_with_retries
                _request_with_retries(request, label="test request", max_retries=1)
        self.assertEqual(attempts, 2)
        self.assertEqual(raised.exception.failure_type, "timeout")
        self.assertTrue(raised.exception.transient)
        self.assertEqual(raised.exception.retry_count, 1)

    def test_repair_proposals_use_provider_structured_output(self):
        spec = ModelSpec.model_validate_json(
            (Path(__file__).parents[1] / "examples/closed_loop_pid.json").read_text()
        )
        diagnosis = RepairDiagnosis(summary="Failed checks.", failed_checks=[{
            "name": "final_value_tracking_error", "value": 0.2, "unit": "output units",
            "criterion": "<= 0.05 output units", "reason": "Measured error exceeds limit.",
        }])
        proposal = RepairProposal(
            reason="Reduce measured tracking error.",
            changes_summary="Increase controller P.",
            parameter_updates={"controller": {"P": 3}},
        )

        class Responses:
            def parse(self, **kwargs):
                self.kwargs = kwargs
                return type("Response", (), {"output_parsed": proposal})()

        class Interactions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return type("Response", (), {"output_text": proposal.model_dump_json()})()

        openai_client = type("Client", (), {"responses": Responses()})()
        gemini_client = type("Client", (), {"interactions": Interactions()})()
        self.assertEqual(OpenAIPlanner(provider="openai", client=openai_client).propose_repair(spec, diagnosis), proposal)
        self.assertIs(openai_client.responses.kwargs["text_format"], RepairProposal)
        with self.assertLogs("circuitpilot.planner", level="INFO") as captured:
            self.assertEqual(GeminiPlanner(client=gemini_client).propose_repair(spec, diagnosis), proposal)
        self.assertIn("Gemini repair request started", captured.output[0])
        self.assertIn("Gemini repair request succeeded", captured.output[-1])
        self.assertIn("retry_attempt=1", captured.output[0])
        self.assertNotIn(spec.model_dump_json(), "\n".join(captured.output))
        self.assertEqual(gemini_client.interactions.kwargs["response_format"][0]["schema"], RepairProposal.model_json_schema())
        self.assertIn("parameter_updates", gemini_client.interactions.kwargs["response_format"][0]["schema"]["properties"])

    def test_provider_defaults_and_environment_model_override(self):
        self.assertEqual(get_provider_config("grok").api_key_env, "XAI_API_KEY")
        self.assertEqual(get_provider_config("grok").base_url, "https://api.x.ai/v1")
        self.assertEqual(get_provider_config("grok").default_model, "grok-4.7")
        self.assertEqual(get_provider_config("gemini").api_key_env, "GEMINI_API_KEY")
        self.assertIsNone(get_provider_config("gemini").base_url)
        self.assertEqual(get_provider_config("gemini").default_model, "gemini-3.8-flash")
        with patch.dict(os.environ, {"CIRCUITPILOT_PROVIDER": "grok", "CIRCUITPILOT_MODEL": "grok-test-model"}):
            planner = OpenAIPlanner(client=object())
        self.assertEqual(planner.provider, "grok")
        self.assertEqual(planner.model, "grok-test-model")

    def test_client_construction_uses_provider_key_and_base_url(self):
        constructions = []

        def fake_client(**kwargs):
            constructions.append(kwargs)
            return object()

        create_provider_client(
            "grok", client_factory=fake_client, environ={"XAI_API_KEY": "unit-test-token"}
        )
        create_provider_client(
            "openai", client_factory=fake_client, environ={"OPENAI_API_KEY": "unit-test-token"}
        )
        create_provider_client(
            "gemini", client_factory=fake_client, environ={"GEMINI_API_KEY": "unit-test-token"}
        )
        self.assertEqual(constructions[0], {
            "api_key": "unit-test-token", "base_url": "https://api.x.ai/v1",
            "timeout": 60.0, "max_retries": 0,
        })
        self.assertEqual(constructions[1], {
            "api_key": "unit-test-token", "timeout": 60.0, "max_retries": 0,
        })
        self.assertEqual(constructions[2], {
            "api_key": "unit-test-token",
            "http_options": {"timeout": 60000, "retry_options": {"attempts": 0}},
        })

    def test_gemini_interactions_sdk_translation_disables_retries(self):
        try:
            from google import genai
            from google.genai._gaos.google_genai import _translate_retry_config
        except ImportError:
            self.skipTest("google-genai Interactions SDK is not installed")

        client = create_provider_client(
            "gemini",
            client_factory=genai.Client,
            environ={"GEMINI_API_KEY": "offline-test-placeholder"},
        )
        options = client._api_client._http_options
        self.assertEqual(options.retry_options.attempts, 0)
        translated = _translate_retry_config(options)
        self.assertEqual(translated.max_retries, 0)

    def test_gemini_429_uses_only_bounded_circuitpilot_retries_and_rate_limit_category(self):
        secret = "offline-test-key"

        class Interactions:
            calls = 0

            def create(inner_self, **kwargs):
                inner_self.calls += 1
                error = StructuredAdapterTests.http_error(429, retry_after=9999)
                error.response.text = "rate limited api key " + secret
                raise error

        interactions = Interactions()
        client = type("Client", (), {"interactions": interactions})()
        with patch.dict(os.environ, {"GEMINI_API_KEY": secret}), \
             patch("circuitpilot.planner.time.sleep") as sleep, \
             patch("circuitpilot.planner.random.uniform", return_value=0.1):
            with self.assertRaises(ProviderRequestError) as raised:
                GeminiPlanner(client=client, max_retries=1).plan("private prompt text")
        self.assertEqual(interactions.calls, 2)
        self.assertEqual(raised.exception.retry_count, 1)
        self.assertEqual(raised.exception.error_category, "rate_limit")
        self.assertEqual(raised.exception.status_code, 429)
        self.assertTrue(raised.exception.transient)
        self.assertNotIn(secret, str(raised.exception))
        sleep.assert_called_once_with(30.0)

    def test_missing_key_and_unknown_provider_are_clear(self):
        with self.assertRaisesRegex(PlannerError, "Set XAI_API_KEY"):
            create_provider_client("grok", client_factory=lambda **kwargs: object(), environ={})
        with self.assertRaisesRegex(PlannerError, "Unsupported planner provider"):
            get_provider_config("not-a-provider")

    def test_openai_adapter_requests_typed_structured_output(self):
        graph = (Path(__file__).parents[1] / "examples/closed_loop_pid.json").read_text()
        envelope = PlannerEnvelope(status="ready", graph_json=graph, message="")

        class Responses:
            def parse(self, **kwargs):
                self.kwargs = kwargs
                return type("Response", (), {"output_parsed": envelope})()

        class Client:
            responses = Responses()

        client = Client()
        spec = OpenAIPlanner(client=client).plan("Create the example graph")
        self.assertEqual(spec.name, "ClosedLoopPID")
        self.assertIs(client.responses.kwargs["text_format"], PlannerEnvelope)

    def test_gemini_adapter_uses_interactions_json_schema_and_validates_graph(self):
        graph = (Path(__file__).parents[1] / "examples/closed_loop_pid.json").read_text()
        envelope = PlannerEnvelope(status="ready", graph_json=graph, message="")

        class Interactions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return type("Response", (), {"output_text": envelope.model_dump_json()})()

        class Client:
            interactions = Interactions()

        client = Client()
        planner = GeminiPlanner(model="gemini-test", client=client)
        spec = planner.plan("Create the example graph")
        self.assertEqual(spec.name, "ClosedLoopPID")
        self.assertEqual(client.interactions.kwargs["model"], "gemini-test")
        self.assertEqual(client.interactions.kwargs["api_version"], GEMINI_INTERACTIONS_API_VERSION)
        self.assertEqual(client.interactions.kwargs["input"], "Create the example graph")
        self.assertIn("classical-control graph", client.interactions.kwargs["system_instruction"])
        self.assertEqual(client.interactions.kwargs["response_format"], [{
            "type": "text",
            "mime_type": "application/json",
            "schema": PlannerEnvelope.model_json_schema(),
        }])

    def test_gemini_planning_retries_429_and_5xx_then_validates_graph(self):
        graph = (Path(__file__).parents[1] / "examples/closed_loop_pid.json").read_text()
        envelope = PlannerEnvelope(status="ready", graph_json=graph, message="")
        for status in (429, 503):
            with self.subTest(status=status):
                class Interactions:
                    calls = 0

                    def create(inner_self, **kwargs):
                        inner_self.calls += 1
                        if inner_self.calls == 1:
                            raise StructuredAdapterTests.http_error(status, retry_after=0.25)
                        return type("Response", (), {"output_text": envelope.model_dump_json()})()

                interactions = Interactions()
                client = type("Client", (), {"interactions": interactions})()
                with patch("circuitpilot.planner.time.sleep") as sleep, \
                     patch("circuitpilot.planner.random.uniform", return_value=0.1):
                    spec = GeminiPlanner(client=client, max_retries=1).plan("Create a closed-loop PID")
                self.assertEqual(spec.name, "ClosedLoopPID")
                self.assertEqual(interactions.calls, 2)
                sleep.assert_called_once_with(0.25)

    def test_gemini_planning_5xx_exhaustion_is_bounded_and_redacted(self):
        secret = "offline-test-api-key"
        class Interactions:
            calls = 0

            def create(inner_self, **kwargs):
                inner_self.calls += 1
                error = StructuredAdapterTests.http_error(503)
                error.response.text = "service unavailable " + secret
                raise error

        interactions = Interactions()
        client = type("Client", (), {"interactions": interactions})()
        with patch.dict(os.environ, {"GEMINI_API_KEY": secret}), \
             patch("circuitpilot.planner.time.sleep") as sleep, \
             patch("circuitpilot.planner.random.uniform", return_value=0.1):
            with self.assertRaises(ProviderRequestError) as raised:
                GeminiPlanner(client=client, max_retries=2).plan("prompt content must not be logged")
        error = raised.exception
        self.assertEqual(interactions.calls, 3)
        self.assertEqual(error.retry_count, 2)
        self.assertTrue(error.transient)
        self.assertEqual(error.status_code, 503)
        self.assertNotIn(secret, str(error))
        self.assertEqual(sleep.call_count, 2)

    def test_gemini_planning_retries_timeout_with_bounded_attempt_count(self):
        graph = (Path(__file__).parents[1] / "examples/closed_loop_pid.json").read_text()
        envelope = PlannerEnvelope(status="ready", graph_json=graph, message="")

        class Interactions:
            calls = 0

            def create(inner_self, **kwargs):
                inner_self.calls += 1
                if inner_self.calls == 1:
                    raise TimeoutError("mock timeout")
                return type("Response", (), {"output_text": envelope.model_dump_json()})()

        interactions = Interactions()
        client = type("Client", (), {"interactions": interactions})()
        with patch("circuitpilot.planner.time.sleep") as sleep, \
             patch("circuitpilot.planner.random.uniform", return_value=0.1):
            spec = GeminiPlanner(client=client, max_retries=1).plan("Create a closed-loop PID")
        self.assertEqual(spec.name, "ClosedLoopPID")
        self.assertEqual(interactions.calls, 2)
        sleep.assert_called_once()

    def test_invalid_gemini_graph_is_rejected_before_cli_calls_generator(self):
        graph = json.loads((Path(__file__).parents[1] / "examples/closed_loop_pid.json").read_text())
        graph["components"][0]["type"] = "MOSFET"
        envelope = PlannerEnvelope(status="ready", graph_json=json.dumps(graph), message="")

        class Interactions:
            def create(self, **kwargs):
                return type("Response", (), {"output_text": envelope.model_dump_json()})()

        planner = GeminiPlanner(client=type("Client", (), {"interactions": Interactions()})())
        with tempfile.TemporaryDirectory() as temp:
            stderr = io.StringIO()
            with patch("sys.argv", ["circuitpilot", "--prompt", "Create a model", "--provider", "gemini", "--output", temp]), \
                 patch("circuitpilot.cli.GeminiPlanner", return_value=planner), \
                 patch("circuitpilot.cli.generate_and_simulate") as generate, \
                 redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    main()
        self.assertEqual(raised.exception.code, 2)
        generate.assert_not_called()
        self.assertIn("Unsupported component type 'MOSFET'", stderr.getvalue())
        self.assertIn("Supported types: Step, Sum, Gain, Transfer Fcn", stderr.getvalue())

    def test_gemini_request_error_includes_status_and_sanitized_response(self):
        api_key = "test-secret-api-key"

        class FakeResponse:
            status_code = 400
            text = "<html><title>Invalid request</title><body>bad model test-secret-api-key</body></html>"

        class FakeRequestError(Exception):
            status_code = 400
            response = FakeResponse()

        class Interactions:
            def create(self, **kwargs):
                raise FakeRequestError("request rejected")

        class Client:
            interactions = Interactions()

        with patch.dict(os.environ, {"GEMINI_API_KEY": api_key}):
            with self.assertRaises(PlannerError) as raised:
                GeminiPlanner(client=Client()).plan("Create a model")
        message = str(raised.exception)
        self.assertIn("HTTP 400", message)
        self.assertIn("Invalid request", message)
        self.assertIn("[REDACTED]", message)
        self.assertNotIn(api_key, message)
        self.assertNotIn("<html>", message)

    def test_gemini_adapter_rejects_invalid_structured_output(self):
        class Interactions:
            def create(self, **kwargs):
                return type("Response", (), {"output_text": '{"status":"ready"}'})()

        class Client:
            interactions = Interactions()

        with self.assertRaisesRegex(PlannerError, "malformed structured output"):
            GeminiPlanner(client=Client()).plan("Create a model")

    def test_gemini_diagnostic_runs_plain_minimal_and_full_schema_requests(self):
        class Interactions:
            def create(self, **kwargs):
                self.requests.append(kwargs)
                if "response_format" not in kwargs:
                    return type("Response", (), {"output_text": "OK"})()
                schema = kwargs["response_format"][0]["schema"]
                if "message" in schema["properties"]:
                    return type("Response", (), {"output_text": '{"status":"clarification","graph_json":"","message":"Please specify the plant transfer function."}'})()
                return type("Response", (), {"output_text": '{"status":"ok"}'})()

            requests = []

        class Client:
            interactions = Interactions()

        client = Client()
        planner = GeminiPlanner(model="gemini-diagnostic-test", client=client)
        results = planner.diagnose_api()
        self.assertEqual([result["probe"] for result in results], [
            "interactions_plain_text", "interactions_structured_json", "interactions_circuitpilot_schema"
        ])
        self.assertTrue(all(result["success"] for result in results))
        self.assertEqual(client.interactions.requests[0], {
            "model": "gemini-diagnostic-test",
            "input": "Reply with OK.",
            "api_version": GEMINI_INTERACTIONS_API_VERSION,
        })
        small_request = client.interactions.requests[1]
        self.assertEqual(small_request["model"], "gemini-diagnostic-test")
        self.assertEqual(small_request["api_version"], GEMINI_INTERACTIONS_API_VERSION)
        self.assertIsInstance(small_request["response_format"], list)
        self.assertEqual(small_request["response_format"][0]["schema"]["properties"], {"status": {"title": "Status", "type": "string"}})
        self.assertEqual(small_request["system_instruction"], SYSTEM_INSTRUCTIONS)
        self.assertEqual(client.interactions.requests[2]["response_format"][0]["schema"], PlannerEnvelope.model_json_schema())
        self.assertEqual(client.interactions.requests[2]["api_version"], GEMINI_INTERACTIONS_API_VERSION)

    def test_gemini_interactions_endpoint_tracks_explicit_api_version(self):
        class Client:
            _api_client = type("ApiClient", (), {
                "_http_options": type("HttpOptions", (), {
                    "base_url": "https://generativelanguage.googleapis.com/",
                    "api_version": "v1beta",
                })()
            })()

        self.assertEqual(
            GeminiPlanner(client=Client()).api_endpoint,
            "https://generativelanguage.googleapis.com/v1beta",
        )

    def test_gemini_diagnostic_reports_endpoint_and_redacts_api_key(self):
        api_key = "never-print-this-key"

        class Request:
            url = "https://generativelanguage.googleapis.com/v1beta/interactions?key=never-print-this-key"
            method = "POST"

        class FakeResponse:
            status_code = 400
            request = Request()
            headers = {"content-type": "text/html", "date": "today", "x-goog-api-key": api_key}
            text = "<html><title>Error 400</title>schema rejected</html>"

        class FakeApiError(Exception):
            code = 400
            response = FakeResponse()

        class Interactions:
            def create(self, **kwargs):
                raise FakeApiError("request rejected")

        class Client:
            interactions = Interactions()
            models = type("Models", (), {"generate_content": lambda self, **kwargs: (_ for _ in ()).throw(FakeApiError("request rejected"))})()

        with patch.dict(os.environ, {"GEMINI_API_KEY": api_key}):
            results = GeminiPlanner(client=Client()).diagnose_api()
        message = results[0]["detail"]
        self.assertIn("HTTP 400", message)
        self.assertIn("POST https://generativelanguage.googleapis.com/v1beta/interactions", message)
        self.assertIn("HTML response body", message)
        self.assertIn("schema rejected", message)
        self.assertIn("content-type=text/html", message)
        self.assertNotIn(api_key, message)
        self.assertNotIn("?key=", message)
        self.assertNotIn("x-goog-api-key", message)
        self.assertEqual([result["probe"] for result in results], [
            "interactions_plain_text", "interactions_structured_json", "generate_content_plain_text"
        ])

    def test_gemini_diagnostic_falls_back_to_generate_content_after_both_interactions_fail(self):
        class Interactions:
            def create(self, **kwargs):
                raise RuntimeError("Interactions unavailable")

        class Models:
            def generate_content(self, **kwargs):
                self.kwargs = kwargs
                return type("Response", (), {"text": "OK"})()

        class Client:
            interactions = Interactions()
            models = Models()

        client = Client()
        results = GeminiPlanner(model="gemini-diagnostic-test", client=client).diagnose_api()
        self.assertEqual([result["probe"] for result in results], [
            "interactions_plain_text", "interactions_structured_json", "generate_content_plain_text"
        ])
        self.assertFalse(results[0]["success"])
        self.assertFalse(results[1]["success"])
        self.assertTrue(results[2]["success"])
        self.assertEqual(client.models.kwargs, {
            "model": "gemini-diagnostic-test", "contents": "Reply with OK."
        })


class DotEnvLoadingTests(unittest.TestCase):
    def test_loads_project_root_env_independent_of_current_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "project"
            elsewhere = Path(temp) / "elsewhere"
            root.mkdir()
            elsewhere.mkdir()
            (root / ".env").write_text("CIRCUITPILOT_TEST_ROOT=project-value\n", encoding="utf-8")
            (elsewhere / ".env").write_text("CIRCUITPILOT_TEST_CWD=wrong-value\n", encoding="utf-8")
            original_cwd = Path.cwd()
            try:
                with patch("circuitpilot.planner.PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("CIRCUITPILOT_TEST_ROOT", None)
                    os.environ.pop("CIRCUITPILOT_TEST_CWD", None)
                    os.chdir(elsewhere)
                    self.assertTrue(load_project_dotenv())
                    self.assertEqual(os.environ.get("CIRCUITPILOT_TEST_ROOT"), "project-value")
                    self.assertIsNone(os.environ.get("CIRCUITPILOT_TEST_CWD"))
            finally:
                os.chdir(original_cwd)

    def test_existing_environment_value_takes_precedence_over_dotenv(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".env").write_text("CIRCUITPILOT_TEST_PRECEDENCE=file-value\n", encoding="utf-8")
            with patch.dict(os.environ, {"CIRCUITPILOT_TEST_PRECEDENCE": "process-value"}):
                load_project_dotenv(root)
                self.assertEqual(os.environ["CIRCUITPILOT_TEST_PRECEDENCE"], "process-value")


if __name__ == "__main__":
    unittest.main()
