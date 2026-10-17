import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createIdempotencyKey, startRun, startAndRememberRun, getRun, getRunReport, downloadModel, openModelInMatlab, rememberRunId, restoreMostRecentRun, restoreRunById, createRunSubmissionGate, RECENT_RUN_STORAGE_KEY } from './api.js'

const response = (body, status = 200) => ({ ok: status >= 200 && status < 300, status, json: async () => body })

test('startRun posts prompt and duration with an idempotency header, never credentials', async () => {
  let request
  const result = await startRun({ prompt: 'unit step', simulation_duration_seconds: 12, idempotency_key: 'run-once-1' }, async (url, options) => {
    request = { url, options }
    return response({ run_id: 'r1', status: 'queued' }, 202)
  })
  assert.equal(request.url.endsWith('/api/runs'), true)
  assert.deepEqual(JSON.parse(request.options.body), { prompt: 'unit step', simulation_duration_seconds: 12, auto_repair: false })
  assert.equal(request.options.headers['Idempotency-Key'], 'run-once-1')
  assert.equal('api_key' in JSON.parse(request.options.body), false)
  assert.equal(result.run_id, 'r1')
  assert.match(createIdempotencyKey(), /^[A-Za-z0-9._:-]{1,128}$/)
})

test('automatic repair is sent only as an explicit boolean opt-in', async () => {
  let body
  await startRun({ prompt: 'PID', simulation_duration_seconds: 8, auto_repair: true }, async (_url, options) => {
    body = JSON.parse(options.body)
    return response({ run_id: 'r2', status: 'queued', auto_repair: true }, 202)
  })
  assert.equal(body.auto_repair, true)
  assert.equal('provider' in body, false)
})

test('status and report requests use the generated run id', async () => {
  const paths = []
  const fakeFetch = async (url) => { paths.push(url); return response({ status: 'completed' }) }
  await getRun('r 1', fakeFetch)
  await getRunReport('r 1', fakeFetch)
  assert.match(paths[0], /api\/runs\/r%201$/)
  assert.match(paths[1], /api\/runs\/r%201\/report$/)
})

test('refresh restoration reads the saved run id and fetches its status and real report', async () => {
  const payload = JSON.parse(readFileSync(new URL('./fixtures/completed-run.json', import.meta.url), 'utf8'))
  const runId = payload.report.run_id
  const values = new Map()
  const storage = {
    getItem: key => values.get(key) || null,
    setItem: (key, value) => values.set(key, value),
  }
  assert.equal(rememberRunId(runId, storage), true)
  assert.equal(values.has(RECENT_RUN_STORAGE_KEY), true)
  const requests = []
  const restored = await restoreMostRecentRun(storage, async url => {
    requests.push(url)
    return response(url.endsWith('/report') ? payload : { run_id: runId, status: 'completed', stage: 'completed' })
  })
  assert.deepEqual(requests.map(url => url.split('/').slice(-1)[0]), [runId, 'report'])
  assert.equal(restored.run.status, 'completed')
  assert.equal(restored.report.simulation.time.length, 57)
  assert.equal(restored.report.simulation.output.length, 57)
  assert.equal(restored.report.report.verification.checks.peak_overshoot.status, 'fail')
  assert.equal(restored.reportError, null)
})

test('an older run absent from browser storage can be restored by ID without submitting a run', async () => {
  const payload = JSON.parse(readFileSync(new URL('./fixtures/completed-run.json', import.meta.url), 'utf8'))
  const runId = payload.report.run_id
  const values = new Map()
  const storage = { getItem: key => values.get(key) || null, setItem: (key, value) => values.set(key, value) }
  const requests = []
  const fakeFetch = async (url, options = {}) => {
    requests.push({ url, method: options.method || 'GET' })
    return response(url.endsWith('/report')
      ? payload
      : { run_id: runId, status: 'completed', stage: 'completed' })
  }

  assert.equal(await restoreMostRecentRun(storage, fakeFetch), null)
  const restored = await restoreRunById(runId, storage, fakeFetch)

  assert.equal(restored.report.simulation.time.length, 57)
  assert.equal(restored.report.simulation.output.length, 57)
  assert.equal(restored.report.report.verification.status, 'fail')
  assert.deepEqual(requests.map(({ method }) => method), ['GET', 'GET'])
  assert.match(requests[0].url, new RegExp(`/api/runs/${runId}$`))
  assert.match(requests[1].url, new RegExp(`/api/runs/${runId}/report$`))
  assert.equal(values.get(RECENT_RUN_STORAGE_KEY), runId)
})

test('manual existing-run selection rejects path-like IDs before making a request', async () => {
  let requested = false
  await assert.rejects(restoreRunById('../other', { setItem() {} }, async () => { requested = true }), /run ID is invalid/)
  assert.equal(requested, false)
})

test('empty, inaccessible, or stale saved run IDs are handled without a fake success', async () => {
  assert.equal(await restoreMostRecentRun({ getItem: () => null }, async () => { throw new Error('should not fetch') }), null)
  await assert.rejects(restoreMostRecentRun({ getItem: () => { throw new Error('blocked') } }, async () => {}), /browser storage/)
  await assert.rejects(restoreMostRecentRun({ getItem: () => 'missing-run' }, async () => response({ detail: 'Unknown run ID.' }, 404)), /Unknown run ID/)
  await assert.rejects(restoreMostRecentRun({ getItem: () => 'saved' }, async () => { throw new Error('offline') }), /backend is unreachable/)
})

test('restoration reports missing saved report data without fabricating a run failure', async () => {
  const storage = { getItem: () => 'known-run' }
  const restored = await restoreMostRecentRun(storage, async url => response(
    url.endsWith('/report') ? { detail: 'temporarily unavailable' } : { run_id: 'known-run', status: 'completed', stage: 'completed' },
    url.endsWith('/report') ? 503 : 200,
  ))
  assert.equal(restored.run.status, 'completed')
  assert.equal(restored.report, null)
  assert.match(restored.reportError.message, /temporarily unavailable/)
})

test('restoration preserves completed, interrupted, failed, and unavailable states separately', async () => {
  const fixture = JSON.parse(readFileSync(new URL('./fixtures/completed-run.json', import.meta.url), 'utf8'))
  const runId = fixture.report.run_id
  for (const status of ['completed', 'interrupted', 'failed']) {
    const restored = await restoreMostRecentRun({ getItem: () => runId }, async url => response(
      url.endsWith('/report') ? fixture : { run_id: runId, status, stage: status },
    ))
    assert.equal(restored.run.status, status)
    assert.equal(restored.report.report.verification.status, 'fail')
    assert.equal(restored.reportError, null)
  }

  const unavailable = await restoreMostRecentRun({ getItem: () => runId }, async url => response(
    url.endsWith('/report') ? { detail: 'Report unavailable.' } : { run_id: runId, status: 'interrupted', stage: 'interrupted' },
    url.endsWith('/report') ? 404 : 200,
  ))
  assert.equal(unavailable.run.status, 'interrupted')
  assert.equal(unavailable.report, null)
  assert.match(unavailable.reportError.message, /Report unavailable/)
})

test('malformed status and report payloads never become a restored success', async () => {
  const storage = { getItem: () => 'saved' }
  await assert.rejects(restoreMostRecentRun(storage, async () => response({ run_id: 'other', status: 'completed', stage: 'completed' })), /status response is malformed/)
  const restored = await restoreMostRecentRun(storage, async url => response(url.endsWith('/report')
    ? { report: {}, simulation: [], model: null, report_files: {} }
    : { run_id: 'saved', status: 'completed', stage: 'completed' }))
  assert.equal(restored.run.status, 'completed')
  assert.equal(restored.report, null)
  assert.match(restored.reportError.message, /report response is malformed/)
})

test('a run ID is remembered only after an accepted run response', async () => {
  const values = new Map()
  const storage = { setItem: (key, value) => values.set(key, value) }
  await assert.rejects(startAndRememberRun({ prompt: 'x', simulation_duration_seconds: 1 }, storage,
    async () => response({ detail: 'queue full' }, 429)), /queue full/)
  assert.equal(values.has(RECENT_RUN_STORAGE_KEY), false)
  const accepted = await startAndRememberRun({ prompt: 'x', simulation_duration_seconds: 1 }, storage,
    async () => response({ run_id: 'accepted', status: 'queued' }, 202))
  assert.equal(accepted.run_id_persisted, true)
  assert.equal(values.get(RECENT_RUN_STORAGE_KEY), 'accepted')
})

test('the submission gate prevents overlapping duplicate submissions', async () => {
  const gate = createRunSubmissionGate()
  let submissions = 0
  let finish
  const pending = gate(async () => {
    submissions += 1
    await new Promise(resolve => { finish = resolve })
    return 'accepted'
  })
  assert.equal(await gate(async () => { submissions += 1 }), null)
  finish()
  assert.equal(await pending, 'accepted')
  assert.equal(submissions, 1)
})

test('429, validation and unreachable backend errors are actionable', async () => {
  await assert.rejects(startRun({ prompt: 'x', simulation_duration_seconds: 1 }, async () => response({ detail: 'queue full' }, 429)), /queue full/)
  await assert.rejects(startRun({ prompt: 'x', simulation_duration_seconds: 1 }, async () => response({ detail: [{ msg: 'duration invalid' }] }, 422)), /duration invalid/)
  await assert.rejects(startRun({ prompt: 'x', simulation_duration_seconds: 1 }, async () => { throw new Error('network') }), /backend is unreachable/)
})

test('model download requests the run-scoped artifact and reports server failures', async () => {
  let requestedUrl
  const result = await downloadModel('run-1', 0, async url => {
    requestedUrl = url
    return { ok: true, status: 200, headers: { get: name => name === 'content-disposition' ? 'attachment; filename="pid_control_system.slx"' : null }, blob: async () => ({ text: async () => 'model' }) }
  })
  assert.match(requestedUrl, /\/api\/runs\/run-1\/model\?attempt=0$/)
  assert.equal(result.filename, 'pid_control_system.slx')
  assert.equal(await result.blob.text(), 'model')
  await assert.rejects(downloadModel('run-1', null, async () => response({ detail: 'No saved Simulink model is available.' }, 404)), /No saved Simulink model/)
})

test('MATLAB launch action uses only the run endpoint and surfaces local launch errors', async () => {
  let request
  const result = await openModelInMatlab('run-2', async (url, options) => {
    request = { url, options }
    return response({ status: 'launched', message: 'MATLAB launch requested.' })
  })
  assert.match(request.url, /\/api\/runs\/run-2\/open-in-matlab$/)
  assert.equal(request.options.method, 'POST')
  assert.equal(request.options.body, undefined)
  assert.equal(result.status, 'launched')
  await assert.rejects(openModelInMatlab('run-2', async () => response({ detail: 'MATLAB was not found.' }, 503)), /MATLAB was not found/)
})
