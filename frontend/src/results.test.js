import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { buildResponsePlotPoints, deriveDashboardResult, deriveModelArtifact, deriveRepairHistory } from './results.js'

const completedRun = JSON.parse(readFileSync(new URL('./fixtures/completed-run.json', import.meta.url), 'utf8'))
const repairedRun = JSON.parse(readFileSync(new URL('./fixtures/repaired-run.json', import.meta.url), 'utf8'))

test('repair dashboard selects the best passing attempt and its real response', () => {
  const result = deriveDashboardResult(repairedRun)
  assert.equal(repairedRun.report.run_id, '20261017T151302022129Z-4e2afa15')
  assert.equal(result.lifecycleStatus, 'completed')
  assert.equal(result.selectedAttempt, 1)
  assert.equal(result.simulationStatus, 'succeeded')
  assert.equal(result.verificationStatus, 'pass')
  assert.equal(result.checks.final_value_tracking_error.status, 'pass')
  assert.ok(Math.abs(result.checks.final_value_tracking_error.value - 0.000021610610845157474) < 1e-15)
  assert.equal(result.checks.peak_overshoot.status, 'pass')
  assert.equal(result.checks.peak_overshoot.value, 0)
  assert.equal(result.checks.settling_time.status, 'pass')
  assert.ok(Math.abs(result.checks.settling_time.value - 3.9520762510857246) < 1e-12)
  assert.equal(result.simulation.time.length, 311)
  assert.equal(result.simulation.output.length, 311)
  assert.equal(buildResponsePlotPoints(result.simulation).split(' ').length, 311)

  const history = deriveRepairHistory(repairedRun)
  assert.equal(history.bestAttempt, 1)
  assert.equal(history.attempts[0].verificationStatus, 'fail')
  assert.equal(history.attempts[1].verificationStatus, 'pass')
  assert.equal(history.attempts[1].changes_summary, "Set derivative gain D to 1 on the PID controller component 'pid'.")
  assert.equal(history.attempts[0].modelFilename, 'pid_control_system.slx')
  const model = deriveModelArtifact(repairedRun, repairedRun.report.run_id)
  assert.equal(model.filename, 'pid_control_system.slx')
  assert.equal(model.attemptNumber, 1)
  assert.equal(model.label, 'Repaired · attempt 1')
  assert.equal(model.verificationStatus, 'pass')
  assert.equal(model.downloadUrl, '/api/runs/20261017T151302022129Z-4e2afa15/model')
})

test('dashboard exposes the generated model and both file actions', () => {
  const appSource = readFileSync(new URL('./App.jsx', import.meta.url), 'utf8')
  assert.match(appSource, /aria-label="Generated Simulink Model"/)
  assert.match(appSource, /Download \.slx/)
  assert.match(appSource, /Open in MATLAB/)
  assert.match(appSource, /Download original \.slx/)
})

test('repair selection does not infer pass from missing or malformed selected verification', () => {
  const payload = JSON.parse(JSON.stringify(repairedRun))
  payload.repair.attempts[1].verification = { status: 'pass', checks: 'invalid' }
  payload.report.verification = { status: 'pass' }
  const result = deriveDashboardResult(payload)
  assert.equal(result.verificationStatus, 'inconclusive')
  assert.deepEqual(result.checks, {})
  assert.equal(result.simulationStatus, 'succeeded')
})

test('saved real API report fixture maps all 57 samples, checks, and chart coordinates', () => {
  assert.deepEqual(Object.keys(completedRun).sort(), ['model', 'report', 'report_files', 'simulation'])
  const result = deriveDashboardResult(completedRun)
  assert.equal(result.simulation.time.length, 57)
  assert.equal(result.simulation.output.length, 57)
  assert.equal(result.simulation.time.length, result.simulation.output.length)
  assert.ok(result.simulation.time.every(Number.isFinite))
  assert.ok(result.simulation.output.every(Number.isFinite))
  assert.equal(result.simulationStatus, 'succeeded')
  assert.equal(result.verificationStatus, 'fail')
  assert.equal(result.checks.final_value_tracking_error.status, 'pass')
  assert.ok(Math.abs(result.checks.final_value_tracking_error.value - 0.00278313088423887) < 1e-15)
  assert.equal(result.checks.peak_overshoot.status, 'fail')
  assert.ok(Math.abs(result.checks.peak_overshoot.value - 6.38492212769288) < 1e-12)
  assert.equal(result.checks.settling_time.status, 'fail')
  assert.ok(Math.abs(result.checks.settling_time.value - 6.07847133212036) < 1e-12)
  assert.equal(result.failedChecks.length, 2)
  const points = buildResponsePlotPoints(result.simulation).split(' ')
  assert.equal(points.length, 57)
  assert.ok(points.every(point => point.split(',').length === 2 && point.split(',').every(value => Number.isFinite(Number(value)))))
})

test('successful execution with passing criteria remains distinct and passes', () => {
  const result = deriveDashboardResult({
    report: { simulation_status: 'succeeded', verification: { status: 'pass', checks: { peak_overshoot: { status: 'pass', value: 2, criterion: '<= 5%', reason: 'Within limit.' } } } },
    simulation: { time: [0, 1], output: [0, 1] },
  })
  assert.equal(result.simulationStatus, 'succeeded')
  assert.equal(result.verificationStatus, 'pass')
  assert.equal(result.failedChecks.length, 0)
  assert.deepEqual(result.simulation.output, [0, 1])
})

test('successful simulation with failed engineering criteria exposes failure reasons', () => {
  const simulation = { time: [0, 5, 10], output: [0, 1.063849, 0.996] }
  const result = deriveDashboardResult({
    report: { simulation_status: 'succeeded', verification: { status: 'fail', checks: {
      final_value_tracking_error: { status: 'pass', value: 0.002783, criterion: '<= 0.01 output units', reason: 'Within limit.' },
      peak_overshoot: { status: 'fail', value: 6.3849, criterion: '<= 5%', reason: 'Peak overshoot exceeds the specified limit.' },
      settling_time: { status: 'fail', value: 6.0785, criterion: '<= 5 s', reason: 'Settling time exceeds the specified limit.' },
    } } },
    simulation,
  })
  assert.equal(result.simulationStatus, 'succeeded')
  assert.equal(result.verificationStatus, 'fail')
  assert.deepEqual(result.simulation.time, simulation.time)
  assert.deepEqual(result.simulation.output, simulation.output)
  assert.equal(result.checks.final_value_tracking_error.value, 0.002783)
  assert.equal(result.checks.peak_overshoot.criterion, '<= 5%')
  assert.equal(result.failedChecks.length, 2)
  assert.equal(result.failedChecks[1][1].reason, 'Settling time exceeds the specified limit.')
})

test('missing, misaligned, or non-finite samples do not produce a plotted response', () => {
  assert.equal(deriveDashboardResult({ report: { simulation_status: 'succeeded' } }).simulation, null)
  assert.equal(deriveDashboardResult({ report: { simulation_status: 'succeeded' }, simulation: { time: [], output: [] } }).simulation, null)
  assert.equal(deriveDashboardResult({ simulation: { time: [0, 1], output: [0] } }).simulation, null)
  assert.equal(deriveDashboardResult({ simulation: { time: [0, 1], output: [0, NaN] } }).simulation, null)
  assert.equal(deriveDashboardResult({}).verificationStatus, 'inconclusive')
  assert.equal(buildResponsePlotPoints(undefined), '')
  assert.equal(buildResponsePlotPoints({ time: [], output: [] }), '')
  assert.equal(buildResponsePlotPoints({ time: [0, 1], output: [0, 1] }).split(' ').length, 2)
})

test('repair dashboard data keeps attempt verification and final repair status distinct', () => {
  const history = deriveRepairHistory({ repair: {
    status: 'fail', summary: 'Attempt limit reached', attempt_count: 1, max_repair_attempts: 1, best_attempt: 0,
    attempts: [
      { number: 0, kind: 'original', status: 'fail', changes_summary: 'Original specification', verification: { status: 'fail', checks: { peak_overshoot: { status: 'fail', value: 6.38, unit: '%' } } } },
      { number: 1, kind: 'repair', status: 'fail', changes_summary: 'Tune PID', verification: { status: 'fail', checks: { peak_overshoot: { status: 'pass', value: 4, unit: '%' }, settling_time: { status: 'fail', value: 6, unit: 's' } } } },
    ],
  } })
  assert.equal(history.status, 'fail')
  assert.equal(history.attemptCount, 1)
  assert.equal(history.bestAttempt, 0)
  assert.equal(history.attempts[1].checks.peak_overshoot.value, 4)
  assert.equal(history.attempts[1].verificationStatus, 'fail')
  assert.equal(history.attempts[1].simulationExecutionStatus, 'unknown')
  assert.equal(history.attempts[1].proposalValidation, 'not_applicable')
  assert.equal(deriveRepairHistory({ repair: { enabled: false } }).enabled, false)
  assert.equal(deriveRepairHistory({}), null)
  assert.equal(deriveRepairHistory({ report: { repair: { enabled: true, outcome: 'running', attempt_count: 0, max_attempts: 2 } } }).status, 'running')
})

test('repair history preserves success, failure, interruption, and unavailable states', () => {
  for (const status of ['pass', 'fail', 'interrupted', 'error', 'timed_out']) {
    const history = deriveRepairHistory({ repair: { enabled: true, status, attempts: [
      { number: 1, kind: 'repair', status: 'interrupted', simulation_execution_status: 'interrupted', engineering_verification_status: 'not_run', proposal_validation: 'accepted' },
    ] } })
    assert.equal(history.status, status)
    assert.equal(history.attempts[0].simulationExecutionStatus, 'interrupted')
    assert.equal(history.attempts[0].verificationStatus, 'not_run')
    assert.equal(history.attempts[0].proposalValidation, 'accepted')
  }
  assert.equal(deriveRepairHistory({ repair: { enabled: true, status: 'unavailable', attempts: 'malformed' } }).attempts.length, 0)
  assert.equal(deriveRepairHistory({ repair: { enabled: true, status: 'unavailable' } }).status, 'unavailable')
  const unavailable = deriveRepairHistory(null, { auto_repair: true, status: 'completed', max_repair_attempts: 2 })
  assert.equal(unavailable.status, 'unavailable')
  assert.match(unavailable.summary, /no repair outcome is inferred/)
  assert.equal(deriveRepairHistory(null, { auto_repair: true, status: 'interrupted' }).status, 'interrupted')
  assert.equal(deriveRepairHistory({}), null)
})
