# CircuitPilot frontend MVP

React + Vite frontend for the local CircuitPilot API. Before a run, the diagram and response areas show explicitly labeled illustrative previews. After a run, the UI displays the API's actual stages, verification report, model graph summary, and simulation samples. The browser never receives provider credentials.

## Requirements

- Node.js 16 or newer
- npm 8 or newer

## Install and start (PowerShell)

From the repository root, install backend API dependencies and frontend dependencies:

```powershell
python -m pip install -e ".[api,llm]"
cd frontend
npm install
npm run dev
```

In a second PowerShell window, start the API from the repository root:

```powershell
python -m uvicorn circuitpilot.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Open Vite's local URL (normally `http://127.0.0.1:5173`). Configure another development origin with `CIRCUITPILOT_CORS_ORIGINS` as a comma-separated allowlist. The API defaults to `artifacts/api_runs/<generated-run-id>/`; set `CIRCUITPILOT_API_OUTPUT` to change its output root and `CIRCUITPILOT_API_QUEUE_CAPACITY` (default 2, includes the active run) to bound accepted work. API submissions never enable automatic repair. `npm test` runs offline API-client tests; `npm run build` creates a production frontend build.
