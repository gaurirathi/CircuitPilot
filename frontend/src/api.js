const configuredBase = typeof import.meta.env !== 'undefined' && import.meta.env.VITE_API_BASE_URL
export const API_BASE_URL = (configuredBase || 'http://127.0.0.1:8000').replace(/\/$/, '')
export const RECENT_RUN_STORAGE_KEY = 'circuitpilot.recentRunId'

async function request(path, options = {}, fetchImpl = fetch) {
  let response
  try {
    response = await fetchImpl(`${API_BASE_URL}${path}`, {
      ...options,
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    })
  } catch {
    throw new Error('CircuitPilot backend is unreachable. Start the API server and try again.')
  }
  let body = {}
  try { body = await response.json() } catch { /* keep safe generic message */ }
  if (!response.ok) {
    if (response.status === 429) throw new Error(body.detail || 'The planner rate-limited this request or the run queue is full. Wait, then retry.')
    if (response.status === 422) throw new Error(body.detail?.[0]?.msg || body.detail || 'The run request is invalid.')
    throw new Error(typeof body.detail === 'string' ? body.detail : `CircuitPilot request failed (HTTP ${response.status}).`)
  }
  return body
}

export function createIdempotencyKey() {
  if (typeof globalThis.crypto?.randomUUID === 'function') return globalThis.crypto.randomUUID()
  return `cp-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export function startRun({ prompt, simulation_duration_seconds, auto_repair = false, idempotency_key }, fetchImpl) {
  return request('/api/runs', {
    method: 'POST',
    body: JSON.stringify({ prompt, simulation_duration_seconds, auto_repair: Boolean(auto_repair) }),
    ...(idempotency_key ? { headers: { 'Idempotency-Key': idempotency_key } } : {}),
  }, fetchImpl)
}

export async function startAndRememberRun(requestBody, storage, fetchImpl) {
  const run = await startRun(requestBody, fetchImpl)
  return { ...run, run_id_persisted: rememberRunId(run.run_id, storage) }
}

export function createRunSubmissionGate() {
  let inFlight = false
  return async function submitOnce(submit) {
    if (inFlight) return null
    inFlight = true
    try { return await submit() } finally { inFlight = false }
  }
}

export function getRun(runId, fetchImpl) {
  return request(`/api/runs/${encodeURIComponent(runId)}`, {}, fetchImpl)
}

export function getRunReport(runId, fetchImpl) {
  return request(`/api/runs/${encodeURIComponent(runId)}/report`, {}, fetchImpl)
}

function apiErrorMessage(body, status) {
  return typeof body?.detail === 'string' ? body.detail : `CircuitPilot request failed (HTTP ${status}).`
}

export async function downloadModel(runId, attemptNumber = null, fetchImpl = fetch) {
  const query = Number.isInteger(attemptNumber) && attemptNumber >= 0 ? `?attempt=${attemptNumber}` : ''
  let response
  try {
    response = await fetchImpl(`${API_BASE_URL}/api/runs/${encodeURIComponent(runId)}/model${query}`)
  } catch {
    throw new Error('CircuitPilot backend is unreachable. Start the API server and try again.')
  }
  if (!response.ok) {
    let body = {}
    try { body = await response.json() } catch { /* use safe generic error */ }
    throw new Error(apiErrorMessage(body, response.status))
  }
  const disposition = response.headers?.get?.('content-disposition') || ''
  const filename = disposition.match(/filename="?([^";]+)"?/i)?.[1] || 'simulink_model.slx'
  return { blob: await response.blob(), filename }
}

export async function openModelInMatlab(runId, fetchImpl = fetch) {
  return request(`/api/runs/${encodeURIComponent(runId)}/open-in-matlab`, { method: 'POST' }, fetchImpl)
}

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function validateRestoredRun(run, expectedId) {
  if (!isRecord(run) || run.run_id !== expectedId || typeof run.status !== 'string' || typeof run.stage !== 'string') {
    throw new Error('The saved run status response is malformed.')
  }
  return run
}

function validateReportPayload(payload, expectedId) {
  const report = payload?.report
  const files = payload?.report_files
  if (!isRecord(payload) || !isRecord(report) || report.run_id !== expectedId
    || typeof report.lifecycle_status !== 'string'
    || !(payload.simulation === null || isRecord(payload.simulation))
    || !(payload.model === null || isRecord(payload.model))
    || !isRecord(files) || typeof files.json !== 'string' || typeof files.markdown !== 'string') {
    throw new Error('The saved run is available, but its verification report response is malformed.')
  }
  return payload
}

export function rememberRunId(runId, storage) {
  try {
    const target = storage || globalThis.localStorage
    if (!runId || !target) return false
    target.setItem(RECENT_RUN_STORAGE_KEY, runId)
    return true
  } catch {
    return false
  }
}

export async function restoreMostRecentRun(storage, fetchImpl) {
  let runId
  try { runId = (storage || globalThis.localStorage)?.getItem(RECENT_RUN_STORAGE_KEY) } catch {
    throw new Error('The saved CircuitPilot run could not be read from browser storage.')
  }
  if (!runId) return null
  return restoreRunById(runId, storage, fetchImpl)
}

/** Restore a specific existing run without creating or submitting a new run. */
export async function restoreRunById(runId, storage, fetchImpl) {
  if (typeof runId !== 'string' || !/^[A-Za-z0-9_-]+$/.test(runId)) {
    throw new Error('The requested run ID is invalid.')
  }
  const run = validateRestoredRun(await getRun(runId, fetchImpl), runId)
  // Save only after the backend confirms this ID belongs to an existing run.
  rememberRunId(runId, storage)
  try {
    const payload = validateReportPayload(await getRunReport(runId, fetchImpl), runId)
    return { run, report: payload, reportError: null }
  } catch (reportError) {
    return { run, report: null, reportError }
  }
}
