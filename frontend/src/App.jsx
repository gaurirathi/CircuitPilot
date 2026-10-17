import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity, AlertCircle, ArrowUpRight, Bell, Box, Check, ChevronDown,
  Clock3, Cpu, FileCode2, Gauge, GitBranch, LayoutDashboard, ListChecks,
  Play, Plus, Settings2, ShieldCheck, SlidersHorizontal, Sparkles, Workflow, Zap,
} from 'lucide-react'
import { API_BASE_URL, createIdempotencyKey, createRunSubmissionGate, downloadModel, getRun, getRunReport, openModelInMatlab, restoreMostRecentRun, restoreRunById, startAndRememberRun } from './api.js'
import { buildResponsePlotPoints, deriveDashboardResult, deriveModelArtifact, deriveRepairHistory } from './results.js'

const examples = [
  { label: 'Closed-loop PID', prompt: 'Build a unit-step unity-feedback PID controller for G(s) = 1 / (s² + 2s + 1). Use P = 2, I = 1, D = 0. Check tracking error ≤ 0.01, overshoot ≤ 5%, and settling time ≤ 5 s.' },
  { label: 'First-order response', prompt: 'Create a unit-step response model for G(s) = 1 / (5s + 1). Simulate for 20 seconds and report the settling time.' },
  { label: 'State-space plant', prompt: 'Model a unity-feedback system with a state-space plant A=[[-2]], B=[[1]], C=[[1]], D=[[0]] and a unit-step input. Simulate for 8 seconds.' },
]
const workflow = [
  { name: 'Planning', icon: Sparkles, detail: 'Prompt → graph' },
  { name: 'Validation', icon: ShieldCheck, detail: 'Schema & topology' },
  { name: 'Generation', icon: Workflow, detail: 'Build Simulink model' },
  { name: 'Simulation', icon: Activity, detail: 'Run time response' },
  { name: 'Verification', icon: ListChecks, detail: 'Check criteria' },
]
const terminal = new Set(['completed', 'failed', 'interrupted'])
const checkLabels = {
  simulation_success: 'Simulation status', response_finite: 'Finite response', sample_data: 'Aligned samples',
  final_value_tracking_error: 'Final tracking error', peak_overshoot: 'Peak overshoot', settling_time: 'Settling time',
}

function StageStrip({ job }) {
  const reportStage = job?.report?.report?.stage
  const current = workflow.findIndex((item) => item.name.toLowerCase() === (job?.stage || reportStage || '').toLowerCase())
  return <section className="stage-strip" aria-label="Workflow stages">{workflow.map(({ name, icon: Icon, detail }, index) => {
    let state = !job ? 'Not run' : index < current ? 'Complete' : index === current && !terminal.has(job.status) ? 'In progress' : 'Pending'
    if (terminal.has(job?.status) && index === current) state = job.status === 'completed' ? 'Complete' : 'Stopped'
    if (job?.status === 'completed') state = 'Complete'
    return <div className={`stage ${state === 'In progress' ? 'stage-active' : ''}`} key={name}><div className="stage-icon"><Icon size={17} /></div><div className="stage-copy"><div className="stage-name">{name}</div><div className="stage-detail">{detail}</div></div><span className="stage-state">{state}</span>{index < workflow.length - 1 && <span className="stage-connector" />}</div>
  })}</section>
}

function ResponsePlot({ simulation, demoMode }) {
  const points = useMemo(() => buildResponsePlotPoints(simulation), [simulation])
  return <div className="plot-wrap">
    {!points && <div className="plot-watermark">{demoMode ? 'DEMO PREVIEW · NO SIMULATION DATA' : 'SIMULATION DATA UNAVAILABLE'}</div>}
    <svg className="plot-svg" viewBox="0 0 740 235" role="img" aria-label={points ? 'Actual simulated response' : demoMode ? 'Labeled demo response plot' : 'Response data unavailable'}>
      {[52, 95, 138, 181].map(y => <line key={`h${y}`} className="grid-line" x1="52" x2="713" y1={y} y2={y} />)}
      {[52, 217, 382, 547, 712].map(x => <line key={`v${x}`} className="grid-line" x1={x} x2={x} y1="20" y2="190" />)}
      <line className="axis-line" x1="52" x2="713" y1="190" y2="190" /><line className="axis-line" x1="52" x2="52" y1="20" y2="190" />
      {points && <polyline points={points} fill="none" stroke="#3b82d0" strokeWidth="3" strokeLinejoin="round" />}
      <text className="axis-label" x="383" y="224" textAnchor="middle">Time (s)</text><text className="axis-label" transform="translate(16 112) rotate(-90)" textAnchor="middle">Output</text>
    </svg>
    <div className="plot-legend"><span><i className="plot-key response-key" />{points ? 'Simulated output' : demoMode ? 'Demo preview only' : 'No real response data'}</span>{simulation?.time?.length > 1 && <span className="plot-note">{simulation.time[0]}–{simulation.time.at(-1)} s · {simulation.time.length} samples</span>}</div>
  </div>
}

function ModelPreview({ model, modelPath, hasRun }) {
  if (!model) return <div className="model-empty">{modelPath ? <>Generated model: <code>{modelPath}</code><br />Component graph details are not available.</> : hasRun ? 'No model graph was returned for this real run.' : 'A validated component graph will appear after planning.'}</div>
  return <div className="model-graph"><div className="model-graph-name">{model.name} <span>{model.components.length} blocks · {model.connections.length} connections</span></div><div className="component-chips">{model.components.map(component => <span key={component.id}><b>{component.type}</b><small>{component.id}</small></span>)}</div><div className="connection-list">{model.connections.map((connection, i) => <span key={i}>{connection.source}:{connection.source_port} → {connection.target}:{connection.target_port}</span>)}</div></div>
}

function App() {
  const [prompt, setPrompt] = useState(examples[0].prompt)
  const [duration, setDuration] = useState('10')
  const [autoRepair, setAutoRepair] = useState(false)
  const [job, setJob] = useState(null)
  const [runReport, setRunReport] = useState(null)
  const [restoreState, setRestoreState] = useState('loading')
  const [notice, setNotice] = useState('')
  const [activeNav, setActiveNav] = useState('Workspace')
  const [submitting, setSubmitting] = useState(false)
  const [modelAction, setModelAction] = useState('')
  const [modelNotice, setModelNotice] = useState('')
  const submitGate = useRef(createRunSubmissionGate())
  const busy = restoreState === 'loading' || submitting || Boolean(job && !terminal.has(job.status))
  const demoMode = !job && restoreState === 'empty'
  const dataTag = runReport ? 'API' : job ? 'NO REPORT' : demoMode ? 'DEMO' : 'NO DATA'
  const report = runReport?.report
  const verification = report?.verification
  const resultView = useMemo(() => deriveDashboardResult(runReport), [runReport])
  const repairHistory = useMemo(() => deriveRepairHistory(runReport, job), [runReport, job])
  const modelArtifact = useMemo(() => deriveModelArtifact(runReport, job?.run_id), [runReport, job?.run_id])
  const simulation = resultView.simulation
  const checks = resultView.checks

  useEffect(() => {
    let stopped = false
    async function restore() {
      try {
        // A run_id query parameter is a read-only local debugging selector.
        // It fetches an existing run; it never submits /api/runs.
        const requestedRunId = new URLSearchParams(window.location.search).get('run_id')
        const restored = requestedRunId
          ? await restoreRunById(requestedRunId)
          : await restoreMostRecentRun()
        if (stopped) return
        if (!restored) {
          setRestoreState('empty')
          return
        }
        setJob(restored.run)
        setAutoRepair(Boolean(restored.run.auto_repair))
        setRunReport(restored.report)
        setRestoreState('ready')
        if (restored.reportError) setNotice(restored.reportError.message)
      } catch (error) {
        if (!stopped) {
          setRestoreState('error')
          setNotice(error.message)
        }
      }
    }
    restore()
    return () => { stopped = true }
  }, [])

  useEffect(() => {
    if (!job?.run_id || terminal.has(job.status)) return undefined
    let stopped = false
    let timer
    const poll = async () => {
      try {
        const next = await getRun(job.run_id)
        if (stopped) return
        setJob(next)
        if (next.auto_repair) {
          try { const progressReport = await getRunReport(job.run_id); if (!stopped) setRunReport(progressReport) } catch { /* status remains authoritative */ }
        }
        if (terminal.has(next.status)) {
          const finalReport = await getRunReport(job.run_id)
          if (!stopped) setRunReport(finalReport)
        } else timer = setTimeout(poll, 900)
      } catch (error) {
        if (!stopped) {
          setNotice(error.message)
          if (!terminal.has(job.status)) timer = setTimeout(poll, 2000)
        }
      }
    }
    timer = setTimeout(poll, 300)
    return () => { stopped = true; clearTimeout(timer) }
  }, [job?.run_id, job?.status])

  function chooseExample(item) { setPrompt(item.prompt); setNotice(`${item.label} prompt loaded.`) }

  async function handleRun() {
    if (submitting) return
    setSubmitting(true)
    setNotice('')
    try {
      const created = await submitGate.current(() => startAndRememberRun({
        prompt,
        simulation_duration_seconds: Number(duration),
        auto_repair: autoRepair,
        idempotency_key: createIdempotencyKey(),
      }))
      if (!created) return
      setRunReport(null)
      setJob(null)
      if (!created.run_id_persisted) setNotice('Run started, but this browser could not save its run ID for refresh recovery.')
      setRestoreState('ready')
      setJob({ run_id: created.run_id, status: created.status, stage: created.status, auto_repair: created.auto_repair, status_url: created.status_url, report_url: created.report_url })
    } catch (error) { setNotice(error.message) }
    finally { setSubmitting(false) }
  }

  async function handleModelDownload(attemptNumber = null) {
    if (!job?.run_id || modelAction) return
    setModelAction('download')
    setModelNotice('')
    try {
      const { blob, filename } = await downloadModel(job.run_id, attemptNumber)
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.setTimeout(() => URL.revokeObjectURL(url), 1000)
      setModelNotice(`Downloaded ${filename}.`)
    } catch (error) {
      setModelNotice(error.message)
    } finally { setModelAction('') }
  }

  async function handleOpenModel() {
    if (!job?.run_id || modelAction) return
    setModelAction('open')
    setModelNotice('')
    try {
      const result = await openModelInMatlab(job.run_id)
      setModelNotice(result.message || 'MATLAB launch requested for the selected model.')
    } catch (error) {
      setModelNotice(error.message)
    } finally { setModelAction('') }
  }

  const metric = (key, fallback = '—') => checks[key]?.value ?? fallback
  const criterionResult = (check) => check?.status ? check.status.toUpperCase() : '—'

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#workspace" aria-label="CircuitPilot home"><span className="brand-mark"><Activity size={21} strokeWidth={2.5} /></span><span className="brand-name">circuit<span>pilot</span><small>CONTROL SYSTEMS LAB</small></span></a>
      <div className="workspace-switcher"><div className="workspace-icon"><Cpu size={17} /></div><div><small>WORKSPACE</small><strong>Control Lab</strong></div><ChevronDown size={15} className="muted-icon" /></div>
      <div className="nav-label">WORKSPACE</div><nav className="side-nav" aria-label="Main navigation">{[{ label: 'Workspace', icon: LayoutDashboard }, { label: 'Model library', icon: Box }, { label: 'Run history', icon: Clock3 }].map(({ label, icon: Icon }) => <button className={`nav-item ${activeNav === label ? 'active' : ''}`} key={label} onClick={() => setActiveNav(label)}><Icon size={17} /><span>{label}</span></button>)}</nav>
      <div className="nav-label tools-label">TOOLS</div><nav className="side-nav"><button className="nav-item" onClick={() => setNotice('Supported components are validated by the backend before MATLAB starts.')}><GitBranch size={17} /><span>Component library</span></button><button className="nav-item" onClick={() => setNotice('Provider credentials are configured on the local backend, not in this browser.')}><Settings2 size={17} /><span>Settings</span></button></nav>
      <div className="sidebar-bottom"><div className="help-card"><span className="help-icon"><ShieldCheck size={16} /></span><div><strong>Validated workflow</strong><p>Specs are checked before model generation.</p></div></div><div className="user-row"><div className="avatar">CP</div><div className="user-meta"><strong>Control Engineer</strong><small>Local workspace</small></div><Settings2 size={16} className="user-settings" /></div></div>
    </aside>
    <main className="main-content" id="workspace">
      <header className="topbar"><div className="breadcrumbs"><span>Projects</span><span className="crumb-sep">/</span><strong>Control systems</strong></div><div className="topbar-right"><span className={`offline-pill ${job ? 'connected-pill' : ''}`}><i /> {job ? `Run ${job.status}` : restoreState === 'error' ? 'Saved run unavailable' : restoreState === 'loading' ? 'Restoring run…' : 'No run loaded'}</span><span className="matlab-pill"><span className="matlab-mark">M</span> MATLAB R2024b target</span><button className="icon-button" aria-label="Notifications" onClick={() => setNotice('No new notifications.')}><Bell size={17} /></button></div></header>
      <div className="page-wrap">
        <section className="page-heading"><div><div className="eyebrow"><span className="eyebrow-line" /> CONTROL MODEL BUILDER</div><h1>Design your control system</h1><p>Describe the behavior you want. CircuitPilot validates the graph before creating a Simulink model.</p></div><button className="new-model-button" onClick={() => { setPrompt(''); setJob(null); setRunReport(null); setAutoRepair(false); setNotice('New prompt started.') }}><Plus size={16} /> New model</button></section>
        {!job && demoMode ? <div className="mock-banner"><span className="mock-banner-icon"><AlertCircle size={17} /></span><div><strong>Demo preview · not a simulation</strong><span>The diagram and empty plot below are illustrative only. Run a request to show real backend results.</span></div><span className="banner-badge">NO RUN YET</span></div> : !job ? <div className={`mock-banner ${restoreState === 'error' ? 'error-banner' : ''}`}><span className="mock-banner-icon"><AlertCircle size={17} /></span><div><strong>{restoreState === 'error' ? 'Saved run could not be restored' : 'Restoring most recent run…'}</strong><span>{restoreState === 'error' ? notice : 'Checking saved run status and report.'}</span></div><span className="banner-badge">{restoreState === 'error' ? 'UNAVAILABLE' : 'LOADING'}</span></div> : <div className={`mock-banner ${job.status === 'failed' || report?.error || (terminal.has(job.status) && !report) ? 'error-banner' : ''}`}><span className="mock-banner-icon">{job.status === 'failed' || report?.error || (terminal.has(job.status) && !report) ? <AlertCircle size={17} /> : <Activity size={17} />}</span><div><strong>{report?.error ? `Run error · ${report.error.category}${report.error.http_status ? ` · HTTP ${report.error.http_status}` : ''}` : terminal.has(job.status) && !report ? 'Run finished · report unavailable' : job.status === 'failed' ? 'Run failed' : `Run ${job.status}`}</strong><span>Run ID {job.run_id} · stage: {job.stage}{report?.error?.phase ? ` · ${report.error.phase}` : ''}{report?.model_path ? ` · ${report.model_path}` : ''}{terminal.has(job.status) && !report && notice ? ` · ${notice}` : ''}</span></div><span className="banner-badge">{job.status.toUpperCase()}</span></div>}
        <section className="composer-card"><div className="composer-head"><div className="composer-title"><span className="composer-spark"><Sparkles size={17} /></span><div><h2>Model requirements</h2><p>Describe the plant, controller, input, and performance targets.</p></div></div><span className="step-counter">01 <i>/</i> 05</span></div>
          <label htmlFor="prompt" className="visually-hidden">Natural-language control-system requirements</label><textarea id="prompt" value={prompt} onChange={event => setPrompt(event.target.value)} placeholder="Example: Build a unity-feedback PID controller for a second-order plant..." disabled={busy} />
          <div className="composer-footer"><div className="example-row"><span className="examples-label">TRY AN EXAMPLE</span>{examples.map(item => <button key={item.label} className="example-chip" onClick={() => chooseExample(item)} disabled={busy}>{item.label}</button>)}</div><div className="duration-control"><label htmlFor="duration"><Clock3 size={14} /> Duration</label><input id="duration" type="number" min="0.1" max="86400" step="0.5" value={duration} onChange={event => setDuration(event.target.value)} disabled={busy} /><span>s</span></div></div>
          <label className="repair-option"><input type="checkbox" checked={autoRepair} onChange={event => setAutoRepair(event.target.checked)} disabled={busy} /> Enable bounded automatic repair <span>(server limit applies; off by default)</span></label>
          <div className="run-row"><span className="keyboard-hint">{busy ? 'MATLAB run in progress · this may take a while' : 'Backend validates before generation'}</span><button className="run-button" onClick={handleRun} disabled={busy || !prompt.trim() || !Number.isFinite(Number(duration)) || Number(duration) <= 0}><Play size={16} fill="currentColor" /> {busy ? 'Running…' : 'Run Simulation'} <span className="button-chevron">→</span></button></div>
          {notice && <div className="notice" role="alert"><AlertCircle size={15} />{notice}<button aria-label="Dismiss notice" onClick={() => setNotice('')}>×</button></div>}
        </section>
        <div className="section-heading workflow-heading"><div><h2>Workflow</h2><p>From natural language to verified response</p></div><span className="not-started-label"><i /> {job ? `Run ${job.status}` : restoreState === 'error' ? 'Restore unavailable' : restoreState === 'loading' ? 'Restoring saved run' : 'Awaiting a run'}</span></div><StageStrip job={job} />
        <div className="section-heading results-heading"><div><h2>Simulation overview <span className="inline-mock">{report ? (terminal.has(job?.status) ? 'LIVE RUN DATA' : 'RUNNING') : demoMode ? 'DEMO STATE' : job && !terminal.has(job.status) ? 'RUNNING' : 'NO REPORT DATA'}</span></h2><p>{report ? `Run ${report.run_id} · ${report.timestamp_utc || ''}` : job ? `Run ${job.run_id} · real report data unavailable` : demoMode ? 'Illustrative demo state only' : 'Restoring saved run'}</p></div><span className="filter-button"><SlidersHorizontal size={15} /> {report?.overall_status?.toUpperCase() || (job ? 'No report' : demoMode ? 'Demo' : 'Unavailable')} <ChevronDown size={14} /></span></div>
        <section className="metrics-grid" aria-label="Simulation and verification metrics">
          {[['Final tracking error', metric('final_value_tracking_error'), 'output units', Gauge], ['Peak overshoot', metric('peak_overshoot'), '%', ArrowUpRight], ['Settling time', metric('settling_time'), 's', Clock3]].map(([label, value, unit, Icon]) => <article className="metric-card" key={label}><div className="metric-top"><span className="metric-icon blue"><Icon size={17} /></span><span className="mock-tag">{dataTag}</span></div><p className="metric-label">{label}</p><div className="metric-value-row"><strong>{value}</strong><span>{unit}</span></div><div className="metric-note">{checks[label === 'Final tracking error' ? 'final_value_tracking_error' : label === 'Peak overshoot' ? 'peak_overshoot' : 'settling_time']?.criterion || (report ? 'No criterion specified' : job ? 'Real result not available' : demoMode ? 'No run data' : 'No report data')}</div></article>)}
          <article className="metric-card simulation-card"><div className="metric-top"><span className="metric-icon green"><Activity size={17} /></span><span className="mock-tag">{dataTag}</span></div><p className="metric-label">Simulation execution</p><div className="sim-status"><strong>{resultView.simulationStatus}</strong></div><div className="metric-note">{report ? `${report.sample_count ?? 0} samples · ${report.simulation_time_range || 'time range unavailable'}` : job ? 'Simulation report unavailable' : demoMode ? 'No solver has run yet' : 'No report data'}</div></article>
          <article className="metric-card simulation-card"><div className="metric-top"><span className={`metric-icon ${resultView.verificationStatus === 'pass' ? 'green' : resultView.verificationStatus === 'fail' || resultView.verificationStatus === 'error' ? 'amber' : 'blue'}`}><ShieldCheck size={17} /></span><span className="mock-tag">{dataTag}</span></div><p className="metric-label">Engineering verification</p><div className="sim-status"><strong>{resultView.verificationStatus.toUpperCase()}</strong></div><div className="metric-note">{resultView.failedChecks.length ? `${resultView.failedChecks.length} required check(s) failed` : verification?.summary || (job ? 'Verification report unavailable' : demoMode ? 'No verification report yet' : 'No report data')}</div></article>
        </section>
        <section className="generated-model-card" aria-label="Generated Simulink Model">
          <div className="generated-model-copy">
            <span className="generated-model-icon"><FileCode2 size={19} /></span>
            <div><div className="generated-model-title">Generated Simulink Model</div>
              {modelArtifact ? <><strong className="generated-model-filename">{modelArtifact.filename}</strong>
                <span className="generated-model-meta">{modelArtifact.label} · final verification {modelArtifact.verificationStatus.toUpperCase()}</span></>
                : <span className="generated-model-meta">{job ? 'No saved .slx model is available for this run.' : 'A saved model will appear after generation.'}</span>}
            </div>
          </div>
          {modelArtifact && <div className="generated-model-actions">
            <button type="button" className="model-action secondary" onClick={() => handleModelDownload()} disabled={Boolean(modelAction)}>{modelAction === 'download' ? 'Downloading…' : 'Download .slx'}</button>
            <button type="button" className="model-action primary" onClick={handleOpenModel} disabled={Boolean(modelAction)}>{modelAction === 'open' ? 'Opening…' : 'Open in MATLAB'}</button>
          </div>}
          {modelNotice && <div className="model-action-notice" role="status">{modelNotice}</div>}
        </section>
        <section className="visual-grid">
          <article className="panel model-panel"><div className="panel-header"><div><div className="panel-title"><Workflow size={16} /><h3>Model graph</h3><span className="mock-tag">{runReport?.model ? 'API' : demoMode ? 'PREVIEW' : 'NO DATA'}</span></div><p>{runReport?.model ? `${runReport.model.components.length} validated components` : demoMode ? 'Illustrative area · no generated model' : 'Model graph unavailable'}</p></div><span className="panel-action"><FileCode2 size={15} />{report?.model_path ? 'SLX saved' : 'Model view'}</span></div><ModelPreview model={runReport?.model} modelPath={report?.model_path} hasRun={Boolean(job)} /></article>
          <article className="panel plot-panel"><div className="panel-header"><div><div className="panel-title"><Activity size={16} /><h3>Step response</h3><span className="mock-tag">{simulation?.output ? 'API' : demoMode ? 'PREVIEW' : 'NO DATA'}</span></div><p>{simulation?.output ? 'Actual solver samples' : demoMode ? 'No simulated response · demo state' : 'No simulated response data was returned'}</p></div></div><ResponsePlot simulation={simulation} demoMode={demoMode} /></article>
        </section>
        <section className="checks-panel"><div className="checks-header"><div><div className="panel-title"><ShieldCheck size={17} /><h3>Verification checks</h3><span className="mock-tag">{verification ? 'API' : 'NO DATA'}</span></div><p>{verification?.summary || 'The verifier report will appear after simulation.'}</p></div><span className="checks-summary">{report?.overall_status?.toUpperCase() || '—'}</span></div>
          <div className="checks-table"><div className="check-row check-head"><span>CRITERION</span><span>MEASURED</span><span>LIMIT</span><span>STATUS</span></div>{Object.entries(checks).length ? Object.entries(checks).map(([key, check]) => <div className="check-row" key={key}><span><i className={`check-indicator ${check.status === 'pass' ? 'green-dot' : check.status === 'fail' ? 'amber-dot' : ''}`} />{checkLabels[key] || key}</span><span className="mono-value">{check.value == null ? '—' : `${check.value}${check.unit ? ` ${check.unit}` : ''}`}</span><span className="mono-value">{check.criterion || 'No limit specified'}</span><span className={`result-pill ${check.status === 'pass' ? 'pass' : check.status === 'fail' ? 'fail' : ''}`}>{check.status === 'pass' ? <Check size={12} /> : <AlertCircle size={12} />}{check.status}</span><span className="check-reason">{check.reason}</span></div>) : <div className="checks-empty">{demoMode ? 'Demo state only; no verification data is being shown as measured.' : job ? 'No verification data was returned for this real run.' : 'Waiting to restore saved run data.'}</div>}</div>
          {report?.next_step && <div className="checks-footnote"><Zap size={14} /> {report.next_step}</div>}
        </section>
        {report && <section className="report-details" aria-label="Run reports"><strong>Persistent report</strong><span>Overall: {report.overall_status?.toUpperCase()} · {report.stage}</span><a href={`${API_BASE_URL}/api/runs/${encodeURIComponent(job.run_id)}/report`} target="_blank" rel="noreferrer">Open JSON report</a><a href={`${API_BASE_URL}/api/runs/${encodeURIComponent(job.run_id)}/report.md`} target="_blank" rel="noreferrer">Open Markdown report</a><span>{report.next_step}</span></section>}
        {repairHistory?.enabled && <section className="repair-history" aria-label="Automatic repair history"><div><h3>Automatic repair {repairHistory.status === 'running' ? 'in progress' : 'history'} <span>{repairHistory.attemptCount}/{repairHistory.maxAttempts} repair attempts</span></h3><p>Outcome: <strong>{repairHistory.status}</strong>{repairHistory.bestAttempt == null ? '' : ` · Best attempt ${repairHistory.bestAttempt}`} · {repairHistory.summary}</p></div><ol>{repairHistory.attempts.map(attempt => <li key={attempt.number}><div className="repair-attempt-heading"><strong>Attempt {attempt.number} · {attempt.kind}</strong><span>{attempt.status.toUpperCase()}</span></div>{attempt.modelFilename && <div className="repair-model-file">Model: <code>{attempt.modelFilename}</code>{attempt.kind === 'original' && <button type="button" className="text-action" onClick={() => handleModelDownload(attempt.number)} disabled={Boolean(modelAction)}>Download original .slx</button>}</div>}<p>{attempt.changes_summary}</p><p className="repair-reason">Proposal: {attempt.proposalValidation} · Simulation: {attempt.simulationExecutionStatus} · Verification: {attempt.verificationStatus}</p>{attempt.failure_reason && <p className="repair-reason">{attempt.failure_reason}</p>}<div className="repair-checks">{Object.entries(attempt.checks).map(([key, check]) => <span key={key} className={check.status}>{checkLabels[key] || key}: {check.status}{check.value == null ? '' : ` · ${check.value}${check.unit ? ` ${check.unit}` : ''}`}{check.criterion ? ` (${check.criterion})` : ''}</span>)}</div></li>)}</ol></section>}
        <footer className="page-footer"><span>CircuitPilot <span>·</span> Control systems workspace</span><span>{job ? `Run ${job.status}` : 'Local API integration'} <i /></span></footer>
      </div>
    </main>
  </div>
}

export default App
