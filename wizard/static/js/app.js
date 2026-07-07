/* ═══════════════════════════════════════════════════════════════════════════
   ANE EXPERIMENT WIZARD — Client-Side JavaScript
   WebSocket connection, wizard navigation, live charts, and UI interactions.
   ═══════════════════════════════════════════════════════════════════════════ */

// ── Global State ─────────────────────────────────────────────────────────
const APP = {
  socket: null,
  currentStep: 1,
  totalSteps: 5,
  experimentRunning: false,
  liveChart: null,
  chartData: { labels: [], cpu: [], ane: [], gpu: [], compute: [] },
  sampleIndex: 0,
};

// ── Preferences (localStorage) ───────────────────────────────────────────
const PREFS_KEY = 'ane_wizard_prefs';

function savePrefs() {
  const prefs = {
    model_id: document.getElementById('model-select')?.value || '',
    prompt_source: document.getElementById('prompt-source')?.value || '',
    eval_mode: document.getElementById('eval-mode')?.value || '',
    metric_profile: document.getElementById('metric-profile')?.value || '',
    compression: document.getElementById('compression')?.value || '',
  };
  localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
}

function loadPrefs() {
  try {
    return JSON.parse(localStorage.getItem(PREFS_KEY)) || {};
  } catch { return {}; }
}

function applyPrefs() {
  const prefs = loadPrefs();
  for (const [key, val] of Object.entries(prefs)) {
    const el = document.getElementById(key === 'model_id' ? 'model-select' :
                                       key === 'prompt_source' ? 'prompt-source' :
                                       key === 'eval_mode' ? 'eval-mode' :
                                       key === 'metric_profile' ? 'metric-profile' :
                                       key === 'compression' ? 'compression' : '');
    if (el && val) el.value = val;
  }
}

// ── WebSocket ────────────────────────────────────────────────────────────
function initSocket() {
  if (typeof io === 'undefined') {
    console.warn('Socket.IO not loaded');
    return;
  }

  APP.socket = io();

  APP.socket.on('connect', () => {
    console.log('Connected to server');
    updateConnectionStatus(true);
  });

  APP.socket.on('disconnect', () => {
    console.log('Disconnected from server');
    updateConnectionStatus(false);
  });

  // Experiment events
  APP.socket.on('progress', handleProgress);
  APP.socket.on('validation', handleValidation);
  APP.socket.on('metric_sample', handleMetricSample);
  APP.socket.on('benchmark_progress', handleBenchmarkProgress);
  APP.socket.on('token', handleToken);
  APP.socket.on('completed', handleCompleted);
  APP.socket.on('error', handleError);
  APP.socket.on('cancelled', handleCancelled);
  APP.socket.on('download_progress', handleDownloadProgress);
}

function updateConnectionStatus(connected) {
  const dot = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  if (dot && text) {
    dot.className = 'status-dot ' + (connected ? 'idle' : 'error');
    text.textContent = connected ? 'Ready' : 'Disconnected';
  }
}

// ── Event Handlers ───────────────────────────────────────────────────────
function handleProgress(data) {
  const pct = Math.round((data.step / data.total) * 100);
  updateProgressBar(pct, data.message);
  addConsoleLog(data.message, 'info');
}

function handleValidation(data) {
  const container = document.getElementById('validation-results');
  if (!container) return;

  container.innerHTML = '';
  data.results.forEach(r => {
    const item = document.createElement('div');
    item.className = 'validation-item';
    item.innerHTML = `
      <div class="validation-icon ${r.passed ? 'pass' : 'fail'}">${r.passed ? '✓' : '✗'}</div>
      <span>${r.name}: ${r.message}</span>
    `;
    container.appendChild(item);

    if (!r.passed && r.fix) {
      const fix = document.createElement('div');
      fix.className = 'validation-fix';
      fix.textContent = '↳ ' + r.fix;
      container.appendChild(fix);
    }
  });

  container.classList.remove('hidden');
}

function handleMetricSample(data) {
  APP.sampleIndex++;
  APP.chartData.labels.push(APP.sampleIndex);
  APP.chartData.cpu.push(data.cpu_mw || 0);
  APP.chartData.ane.push(data.ane_mw || 0);
  APP.chartData.gpu.push(data.gpu_mw || 0);
  APP.chartData.compute.push(data.compute_mw || 0);

  // Keep last 120 points
  if (APP.chartData.labels.length > 120) {
    APP.chartData.labels.shift();
    APP.chartData.cpu.shift();
    APP.chartData.ane.shift();
    APP.chartData.gpu.shift();
    APP.chartData.compute.shift();
  }

  if (APP.liveChart) {
    APP.liveChart.data.labels = APP.chartData.labels;
    APP.liveChart.data.datasets[0].data = APP.chartData.cpu;
    APP.liveChart.data.datasets[1].data = APP.chartData.ane;
    APP.liveChart.data.datasets[2].data = APP.chartData.gpu;
    APP.liveChart.update('none');
  }

  // Update live metrics display
  updateLiveMetric('live-cpu', data.cpu_mw, 'mW');
  updateLiveMetric('live-ane', data.ane_mw, 'mW');
  updateLiveMetric('live-gpu', data.gpu_mw, 'mW');
  updateLiveMetric('live-thermal', data.thermal, '');
}

function handleBenchmarkProgress(data) {
  if (data.completed !== undefined && data.total !== undefined) {
    const pct = Math.round((data.completed / data.total) * 100);
    updateProgressBar(pct, `Benchmark: ${data.completed}/${data.total} prompts completed`);
  }

  if (data.last_result) {
    updateLiveMetric('live-tps', data.last_result.tokens_per_sec, 'tok/s');
    addConsoleLog(
      `Prompt done: ${data.last_result.tokens_generated} tokens @ ${data.last_result.tokens_per_sec} tok/s`,
      'success'
    );
  }
}

function handleToken(data) {
  const output = document.getElementById('live-output');
  if (output) {
    output.textContent += data.token;
    output.scrollTop = output.scrollHeight;
  }
}

function handleCompleted(data) {
  APP.experimentRunning = false;
  showToast('Experiment completed successfully!', 'success');
  addConsoleLog('✓ Experiment completed!', 'success');
  updateProgressBar(100, 'Experiment complete!');
  toggleExperimentButtons(false);

  // Show post-experiment actions
  const actionsEl = document.getElementById('post-actions');
  if (actionsEl) {
    actionsEl.classList.remove('hidden');
    actionsEl.dataset.experimentId = data.experiment_id;
  }

  // Update sidebar status
  updateConnectionStatus(true);
}

function handleError(data) {
  APP.experimentRunning = false;
  showToast(data.message || 'An error occurred', 'error');
  addConsoleLog('✗ Error: ' + (data.message || 'Unknown error'), 'error');

  if (data.details) {
    data.details.forEach(d => addConsoleLog('  → ' + d, 'error'));
  }

  toggleExperimentButtons(false);
}

function handleCancelled() {
  APP.experimentRunning = false;
  showToast('Experiment cancelled', 'warning');
  addConsoleLog('Experiment cancelled by user', 'warning');
  toggleExperimentButtons(false);
}

function handleDownloadProgress(data) {
  addConsoleLog(data.message, 'info');
}

// ── Wizard Navigation ────────────────────────────────────────────────────
function goToStep(step) {
  if (step < 1 || step > APP.totalSteps) return;

  // Validate current step before advancing
  if (step > APP.currentStep && !validateStep(APP.currentStep)) return;

  APP.currentStep = step;

  // Update dots
  document.querySelectorAll('.wizard-dot').forEach((dot, i) => {
    dot.classList.remove('active', 'completed');
    if (i + 1 === step) dot.classList.add('active');
    else if (i + 1 < step) dot.classList.add('completed');
  });

  // Update lines
  document.querySelectorAll('.wizard-line').forEach((line, i) => {
    line.classList.toggle('completed', i + 1 < step);
  });

  // Update panels
  document.querySelectorAll('.wizard-panel').forEach((panel, i) => {
    panel.classList.toggle('active', i + 1 === step);
  });

  // Update buttons
  const backBtn = document.getElementById('wizard-back');
  const nextBtn = document.getElementById('wizard-next');
  const startBtn = document.getElementById('wizard-start');

  if (backBtn) backBtn.classList.toggle('hidden', step === 1);
  if (nextBtn) nextBtn.classList.toggle('hidden', step === APP.totalSteps);
  if (startBtn) startBtn.classList.toggle('hidden', step !== APP.totalSteps);

  // If on review step, populate summary
  if (step === 5) populateReview();

  savePrefs();
}

function nextStep() { goToStep(APP.currentStep + 1); }
function prevStep() { goToStep(APP.currentStep - 1); }

function validateStep(step) {
  switch (step) {
    case 1: {
      const model = document.getElementById('model-select')?.value;
      if (!model) { showToast('Please select a model', 'warning'); return false; }
      return true;
    }
    case 2: {
      const source = document.getElementById('prompt-source')?.value;
      if (source === 'single') {
        const text = document.getElementById('prompt-text')?.value?.trim();
        if (!text) { showToast('Please enter a prompt', 'warning'); return false; }
      }
      return true;
    }
    default:
      return true;
  }
}

function populateReview() {
  const setReview = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val || '—';
  };

  const modelSelect = document.getElementById('model-select');
  const modelText = modelSelect?.options[modelSelect.selectedIndex]?.text || modelSelect?.value || '—';

  setReview('review-model', modelText);
  setReview('review-compression', document.getElementById('compression')?.value || 'int8');
  setReview('review-prompt-source', document.getElementById('prompt-source')?.value || 'single');
  setReview('review-eval-mode', document.getElementById('eval-mode')?.value || 'single');
  setReview('review-metric-profile', document.getElementById('metric-profile')?.value || 'quick');
  setReview('review-max-len', document.getElementById('max-len')?.value || '512');
  setReview('review-warmup-runs', document.getElementById('warmup-runs')?.value || '1');
  setReview('review-benchmark-runs', document.getElementById('benchmark-runs')?.value || '1');
  setReview('review-device', 'Apple Neural Engine (ANE)');

  // Estimate memory
  const modelId = document.getElementById('model-select')?.value;
  const compress = document.getElementById('compression')?.value || 'int8';
  const memEl = document.getElementById('review-memory');
  if (memEl && modelId) {
    fetch(`/api/models/${encodeURIComponent(modelId)}/memory?compression=${compress}`)
      .then(r => r.json())
      .then(d => { memEl.textContent = d.estimate || '—'; })
      .catch(() => { memEl.textContent = '—'; });
  }
}

// ── Start Experiment ─────────────────────────────────────────────────────
function startExperiment() {
  if (APP.experimentRunning) return;

  const config = {
    model_id: document.getElementById('model-select')?.value,
    compression: document.getElementById('compression')?.value || 'int8',
    prompt_source: document.getElementById('prompt-source')?.value || 'single',
    prompt_text: document.getElementById('prompt-text')?.value || '',
    prompt_file: document.getElementById('prompt-file-path')?.value || '',
    prompt_dataset: document.getElementById('prompt-dataset')?.value || 'general',
    eval_mode: document.getElementById('eval-mode')?.value || 'single',
    metric_profile: document.getElementById('metric-profile')?.value || 'quick',
    max_len: parseInt(document.getElementById('max-len')?.value || '512'),
    warmup_runs: parseInt(document.getElementById('warmup-runs')?.value || '1'),
    benchmark_runs: parseInt(document.getElementById('benchmark-runs')?.value || '1'),
    experiment_name: document.getElementById('experiment-name')?.value || '',
    notes: document.getElementById('experiment-notes')?.value || '',
  };

  if (!config.model_id) {
    showToast('No model selected', 'error');
    return;
  }

  APP.experimentRunning = true;
  APP.sampleIndex = 0;
  APP.chartData = { labels: [], cpu: [], ane: [], gpu: [], compute: [] };

  // Navigate to experiment page
  window.location.href = '/experiment';

  // Send start command via API (will redirect)
  fetch('/api/experiment/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  }).then(r => r.json())
    .then(d => {
      if (d.error) {
        showToast(d.error, 'error');
        APP.experimentRunning = false;
      }
    }).catch(e => {
      showToast('Failed to start experiment: ' + e.message, 'error');
      APP.experimentRunning = false;
    });
}

function cancelExperiment() {
  showConfirmDialog(
    'Cancel Experiment',
    'Are you sure you want to cancel the running experiment? Partial data will be lost.',
    () => {
      fetch('/api/experiment/cancel', { method: 'POST' })
        .then(r => r.json())
        .then(() => showToast('Cancelling experiment...', 'warning'));
    }
  );
}

// ── Prompt Source Toggle ─────────────────────────────────────────────────
function onPromptSourceChange() {
  const source = document.getElementById('prompt-source')?.value;
  document.querySelectorAll('.prompt-panel').forEach(p => p.classList.add('hidden'));
  const panel = document.getElementById(`prompt-${source}`);
  if (panel) panel.classList.remove('hidden');
}

// ── Model Search ─────────────────────────────────────────────────────────
function filterModels() {
  const query = document.getElementById('model-search')?.value?.toLowerCase() || '';
  const options = document.getElementById('model-select')?.options;
  if (!options) return;

  for (let i = 0; i < options.length; i++) {
    const text = options[i].text.toLowerCase();
    options[i].style.display = text.includes(query) ? '' : 'none';
  }
}

// ── UI Helpers ───────────────────────────────────────────────────────────
function updateProgressBar(pct, message) {
  const bar = document.getElementById('progress-bar');
  const text = document.getElementById('progress-message');
  const pctText = document.getElementById('progress-pct');

  if (bar) bar.style.width = pct + '%';
  if (text) text.textContent = message || '';
  if (pctText) pctText.textContent = pct + '%';
}

function updateLiveMetric(id, value, unit) {
  const el = document.getElementById(id);
  if (el) {
    if (typeof value === 'number') {
      el.textContent = value.toFixed(1) + ' ' + unit;
    } else {
      el.textContent = (value || '—') + ' ' + unit;
    }
  }
}

function addConsoleLog(message, type = '') {
  const console_el = document.getElementById('live-console');
  if (!console_el) return;

  const entry = document.createElement('div');
  entry.className = 'log-entry ' + type;
  const time = new Date().toLocaleTimeString();
  entry.textContent = `[${time}] ${message}`;
  console_el.appendChild(entry);
  console_el.scrollTop = console_el.scrollHeight;
}

function toggleExperimentButtons(running) {
  const startBtn = document.getElementById('wizard-start');
  const cancelBtn = document.getElementById('cancel-experiment');

  if (startBtn) startBtn.disabled = running;
  if (cancelBtn) cancelBtn.classList.toggle('hidden', !running);
}

// ── Toast Notifications ──────────────────────────────────────────────────
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container') || createToastContainer();
  const toast = document.createElement('div');
  toast.className = 'toast ' + type;
  toast.innerHTML = `<span>${type === 'success' ? '✓' : type === 'error' ? '✗' : type === 'warning' ? '⚠' : 'ℹ'}</span> ${message}`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(40px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

function createToastContainer() {
  const container = document.createElement('div');
  container.id = 'toast-container';
  container.className = 'toast-container';
  document.body.appendChild(container);
  return container;
}

// ── Confirmation Dialog ──────────────────────────────────────────────────
function showConfirmDialog(title, message, onConfirm) {
  const overlay = document.getElementById('dialog-overlay');
  if (!overlay) return onConfirm();

  document.getElementById('dialog-title').textContent = title;
  document.getElementById('dialog-message').textContent = message;
  overlay.classList.add('active');

  const confirmBtn = document.getElementById('dialog-confirm');
  const cancelBtn = document.getElementById('dialog-cancel');

  const cleanup = () => { overlay.classList.remove('active'); };

  confirmBtn.onclick = () => { cleanup(); onConfirm(); };
  cancelBtn.onclick = cleanup;
}

// ── Live Chart ───────────────────────────────────────────────────────────
function initLiveChart() {
  const canvas = document.getElementById('live-chart');
  if (!canvas || typeof Chart === 'undefined') return;

  APP.liveChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'CPU (mW)',
          data: [],
          borderColor: '#60a5fa',
          backgroundColor: 'rgba(96,165,250,0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.3,
          pointRadius: 0,
        },
        {
          label: 'ANE (mW)',
          data: [],
          borderColor: '#a78bfa',
          backgroundColor: 'rgba(167,139,250,0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.3,
          pointRadius: 0,
        },
        {
          label: 'GPU (mW)',
          data: [],
          borderColor: '#34d399',
          backgroundColor: 'rgba(52,211,153,0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.3,
          pointRadius: 0,
        },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: {
          display: true,
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#5e5e78', maxTicksLimit: 10 },
          title: { display: true, text: 'Sample #', color: '#5e5e78' },
        },
        y: {
          display: true,
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#5e5e78' },
          title: { display: true, text: 'Power (mW)', color: '#5e5e78' },
          beginAtZero: true,
        }
      },
      plugins: {
        legend: {
          labels: { color: '#9898b0', usePointStyle: true, pointStyle: 'circle' },
        }
      }
    }
  });
}

// ── Results Chart ────────────────────────────────────────────────────────
function initResultsChart(data) {
  const canvas = document.getElementById('results-chart');
  if (!canvas || typeof Chart === 'undefined' || !data) return;

  new Chart(canvas, {
    type: 'line',
    data: {
      labels: data.labels || [],
      datasets: [
        {
          label: 'CPU (mW)',
          data: data.cpu_mw || [],
          borderColor: '#60a5fa',
          backgroundColor: 'rgba(96,165,250,0.08)',
          borderWidth: 2,
          fill: true,
          tension: 0.3,
          pointRadius: 0,
        },
        {
          label: 'ANE (mW)',
          data: data.ane_mw || [],
          borderColor: '#a78bfa',
          backgroundColor: 'rgba(167,139,250,0.08)',
          borderWidth: 2,
          fill: true,
          tension: 0.3,
          pointRadius: 0,
        },
        {
          label: 'GPU (mW)',
          data: data.gpu_mw || [],
          borderColor: '#34d399',
          backgroundColor: 'rgba(52,211,153,0.08)',
          borderWidth: 2,
          fill: true,
          tension: 0.3,
          pointRadius: 0,
        },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#5e5e78' } },
        y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#5e5e78' }, beginAtZero: true },
      },
      plugins: {
        legend: { labels: { color: '#9898b0', usePointStyle: true } },
      }
    }
  });
}

// ── Compare Chart ────────────────────────────────────────────────────────
function initCompareChart(exp1Data, exp2Data, exp1Name, exp2Name) {
  const canvas = document.getElementById('compare-chart');
  if (!canvas || typeof Chart === 'undefined') return;

  const metrics = ['avg_cpu_mw', 'avg_gpu_mw', 'avg_ane_mw', 'avg_compute_mw'];
  const labels = ['CPU Power', 'GPU Power', 'ANE Power', 'Compute Power'];

  new Chart(canvas, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          label: exp1Name,
          data: metrics.map(k => exp1Data[k] || 0),
          backgroundColor: 'rgba(99,102,241,0.6)',
          borderColor: '#6366f1',
          borderWidth: 1,
          borderRadius: 4,
        },
        {
          label: exp2Name,
          data: metrics.map(k => exp2Data[k] || 0),
          backgroundColor: 'rgba(139,92,246,0.6)',
          borderColor: '#8b5cf6',
          borderWidth: 1,
          borderRadius: 4,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#9898b0' } },
        y: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#9898b0' },
          title: { display: true, text: 'Power (mW)', color: '#9898b0' },
          beginAtZero: true,
        },
      },
      plugins: {
        legend: { labels: { color: '#9898b0', usePointStyle: true } },
      }
    }
  });
}

// ── Notes ────────────────────────────────────────────────────────────────
function saveNotes(experimentId) {
  const notes = document.getElementById('experiment-notes-edit')?.value || '';
  fetch(`/api/experiment/${encodeURIComponent(experimentId)}/notes`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ notes }),
  }).then(r => r.json())
    .then(d => {
      if (d.success) showToast('Notes saved', 'success');
      else showToast('Failed to save notes', 'error');
    });
}

// ── Init ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initSocket();
  applyPrefs();

  // Prompt source change listener
  const promptSource = document.getElementById('prompt-source');
  if (promptSource) promptSource.addEventListener('change', onPromptSourceChange);

  // Model search
  const modelSearch = document.getElementById('model-search');
  if (modelSearch) modelSearch.addEventListener('input', filterModels);

  // Init charts if on relevant pages
  if (document.getElementById('live-chart')) initLiveChart();
});
