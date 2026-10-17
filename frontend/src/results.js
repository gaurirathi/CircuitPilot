/** Derive dashboard display state strictly from the API report and samples. */
export function deriveDashboardResult(payload) {
  const report = payload?.report || null
  // The repair report is the authoritative record of which attempt was selected.
  // Its selected verification can be newer than a stale/incomplete report-level
  // projection, so prefer that exact attempt while retaining the full history.
  const repair = payload?.repair || report?.repair || null
  const attempts = Array.isArray(repair?.attempts) ? repair.attempts : []
  const selectedAttempt = Number.isInteger(repair?.best_attempt)
    ? attempts.find(attempt => attempt?.number === repair.best_attempt) || null
    : null
  const selectedVerification = selectedAttempt?.verification
  const reportVerification = report?.verification
  const isValidVerification = value => value && typeof value === 'object'
    && ['pass', 'fail', 'inconclusive', 'error'].includes(value.status)
    && value.checks && typeof value.checks === 'object' && !Array.isArray(value.checks)
    && Object.values(value.checks).every(check => check && typeof check === 'object'
      && ['pass', 'fail', 'inconclusive'].includes(check.status))
  const verification = isValidVerification(selectedVerification)
    ? selectedVerification
    : isValidVerification(reportVerification) ? reportVerification : null
  const checks = verification?.checks || {}
  const topLevelSimulation = payload?.simulation
  const attemptSimulation = selectedAttempt?.simulation
  const raw = topLevelSimulation?.time && topLevelSimulation?.output
    ? topLevelSimulation : attemptSimulation
  const times = raw?.time
  const outputs = raw?.output
  const hasResponse = Array.isArray(times) && Array.isArray(outputs)
    && times.length >= 2 && times.length === outputs.length
    && times.every(Number.isFinite) && outputs.every(Number.isFinite)
    && times.every((time, index) => index === 0 || time > times[index - 1])
  return {
    report,
    lifecycleStatus: report?.lifecycle_status || 'unavailable',
    selectedAttempt: selectedAttempt?.number ?? null,
    verificationStatus: verification?.status || (report?.error ? 'error' : 'inconclusive'),
    simulationStatus: report?.simulation_status
      || selectedAttempt?.simulation_execution_status
      || raw?.simulation_status || 'not reported',
    checks,
    failedChecks: Object.entries(checks).filter(([, check]) => check?.status === 'fail'),
    simulation: hasResponse ? raw : null,
  }
}

/** Keep repair progress separate from the selected simulation and its verification. */
export function deriveRepairHistory(payload, runStatus = null) {
  const repair = payload?.repair || payload?.report?.repair || null
  if (!repair || typeof repair !== 'object') {
    if (runStatus?.auto_repair) return {
      enabled: true,
      status: runStatus.status === 'interrupted' ? 'interrupted' : 'unavailable',
      summary: 'Repair report is unavailable; no repair outcome is inferred.',
      attemptCount: 0,
      maxAttempts: runStatus.max_repair_attempts ?? 0,
      bestAttempt: null,
      attempts: [],
    }
    return null
  }
  const attempts = Array.isArray(repair.attempts) ? repair.attempts : []
  return {
    enabled: repair.enabled ?? true,
    status: repair.status || repair.outcome || 'unavailable',
    summary: repair.summary || '',
    attemptCount: repair.attempt_count ?? attempts.filter(item => item?.kind === 'repair').length,
    maxAttempts: repair.max_repair_attempts ?? repair.max_attempts ?? 0,
    bestAttempt: repair.best_attempt ?? null,
    attempts: attempts.map(attempt => ({
      ...attempt,
      modelFilename: modelFilename(attempt?.model_path),
      verificationStatus: attempt.engineering_verification_status || attempt.verification?.status || attempt.status,
      simulationExecutionStatus: attempt.simulation_execution_status || 'unknown',
      proposalValidation: attempt.proposal_validation || 'not_applicable',
      checks: attempt.verification?.checks && typeof attempt.verification.checks === 'object'
        ? attempt.verification.checks : {},
    })),
  }
}

function modelFilename(path) {
  return typeof path === 'string' ? path.split(/[\\/]/).filter(Boolean).at(-1) || null : null
}

/** Select display metadata using the same best-attempt record as the API. */
export function deriveModelArtifact(payload, runId) {
  const report = payload?.report || null
  const repair = payload?.repair || report?.repair || null
  const attempts = Array.isArray(repair?.attempts) ? repair.attempts : []
  const selectedAttempt = Number.isInteger(repair?.best_attempt)
    ? attempts.find(attempt => attempt?.number === repair.best_attempt) || null
    : null
  const path = selectedAttempt?.model_path || report?.model_path || payload?.simulation?.model_path || null
  if (!runId || !modelFilename(path)) return null
  const attemptNumber = selectedAttempt?.number ?? null
  const attemptKind = selectedAttempt?.kind || null
  const label = attemptKind === 'original'
    ? `Original · attempt ${attemptNumber}`
    : attemptKind ? `Repaired · attempt ${attemptNumber}` : 'Saved run model'
  return {
    filename: modelFilename(path),
    attemptNumber,
    attemptKind,
    label,
    verificationStatus: selectedAttempt?.verification?.status || report?.verification?.status || 'inconclusive',
    downloadUrl: `/api/runs/${encodeURIComponent(runId)}/model`,
    available: true,
  }
}

/** Return SVG polyline coordinates only for a finite, aligned real response. */
export function buildResponsePlotPoints(simulation) {
  const times = simulation?.time
  const values = simulation?.output
  const valid = Array.isArray(times) && Array.isArray(values)
    && times.length >= 2 && times.length === values.length
    && times.every(Number.isFinite) && values.every(Number.isFinite)
    && times.every((time, index) => index === 0 || time > times[index - 1])
  if (!valid) return ''
  const minY = Math.min(...values), maxY = Math.max(...values)
  const rangeY = maxY - minY || 1
  const minX = times[0], rangeX = times[times.length - 1] - minX || 1
  return values.map((value, i) => `${52 + ((times[i] - minX) / rangeX) * 660},${190 - ((value - minY) / rangeY) * 160}`).join(' ')
}
