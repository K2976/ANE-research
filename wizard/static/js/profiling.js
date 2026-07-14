/* ═══════════════════════════════════════════════════════════════════════════
   LAYER PROFILING — Client-Side JavaScript
   Handles profiling configuration, live telemetry, heatmaps, and charts.
   Completely independent from app.js — no existing code is modified.
   ═══════════════════════════════════════════════════════════════════════════ */

// ── Profiling State ──────────────────────────────────────────────────────
const PROF = {
  runId: null,
  liveChart: null,
  chartData: { labels: [], cpu: [], ane: [], gpu: [] },
  sampleIndex: 0,
  modelStructure: null,
};

// ── Socket Event Handlers ────────────────────────────────────────────────
// These attach to the global APP.socket from app.js

function initProfilingSocket() {
  if (typeof APP === 'undefined' || !APP.socket) {
    // Wait for socket to initialize
    setTimeout(initProfilingSocket, 500);
    return;
  }

  APP.socket.on('profiling_progress', handleProfilingProgress);
  APP.socket.on('profiling_metric_sample', handleProfilingMetric);
  APP.socket.on('profiling_sublayer_energy', handleSublayerEnergy);
  APP.socket.on('profiling_sublayer_accuracy', handleSublayerAccuracy);
  APP.socket.on('profiling_sublayer_complete', handleSublayerComplete);
  APP.socket.on('profiling_completed', handleProfilingCompleted);
  APP.socket.on('profiling_error', handleProfilingError);
  APP.socket.on('profiling_cancelled', handleProfilingCancelled);
  APP.socket.on('profiling_paused', handleProfilingPaused);
  APP.socket.on('profiling_resumed', handleProfilingResumed);
}

function handleProfilingProgress(data, fromPoll) {
  const pct = data.total > 0 ? Math.round((data.step / data.total) * 100) : 0;

  const bar = document.getElementById('prof-progress-bar');
  const msg = document.getElementById('prof-progress-message');
  const pctEl = document.getElementById('prof-progress-pct');

  if (bar) bar.style.width = pct + '%';
  if (msg) msg.textContent = data.message || '';
  if (pctEl) pctEl.textContent = pct + '%';

  // Update state display
  if (data.block !== undefined) setEl('state-block', 'Block ' + data.block);
  if (data.sublayer) setEl('state-sublayer', data.sublayer);
  if (data.bit_width) setEl('state-bitwidth', data.bit_width);
  if (data.phase) setEl('state-phase', data.phase);
  if (data.elapsed_sec !== undefined) setEl('state-elapsed', formatTime(data.elapsed_sec));
  if (data.remaining_sec !== undefined) setEl('state-remaining', '~' + formatTime(data.remaining_sec));
  if (data.energy_records !== undefined) setEl('counter-energy', data.energy_records);
  if (data.accuracy_records !== undefined) setEl('counter-accuracy', data.accuracy_records);

  if (data.telemetry) {
    setEl('telemetry-compile', data.telemetry.avg_compile + 's avg');
    setEl('telemetry-energy', data.telemetry.avg_energy + 's avg');
    setEl('telemetry-accuracy', data.telemetry.avg_accuracy + 's avg');
  }

  // Only add console log for real websocket events, not polling updates
  if (!fromPoll && data.message && data.message !== PROF._lastLoggedMsg) {
    PROF._lastLoggedMsg = data.message;
    addProfilingLog(data.message, 'info');
  }
}

function handleProfilingMetric(data) {
  PROF.sampleIndex++;
  PROF.chartData.labels.push(PROF.sampleIndex);
  PROF.chartData.cpu.push(data.cpu_mw || 0);
  PROF.chartData.ane.push(data.ane_mw || 0);
  PROF.chartData.gpu.push(data.gpu_mw || 0);

  // Keep last 200 points
  if (PROF.chartData.labels.length > 200) {
    PROF.chartData.labels.shift();
    PROF.chartData.cpu.shift();
    PROF.chartData.ane.shift();
    PROF.chartData.gpu.shift();
  }

  if (PROF.liveChart) {
    PROF.liveChart.data.labels = PROF.chartData.labels;
    PROF.liveChart.data.datasets[0].data = PROF.chartData.cpu;
    PROF.liveChart.data.datasets[1].data = PROF.chartData.ane;
    PROF.liveChart.data.datasets[2].data = PROF.chartData.gpu;
    PROF.liveChart.update('none');
  }

  // Update live metrics
  setEl('live-power', (data.compute_mw || 0).toFixed(1) + ' mW');
  setEl('live-temp', data.thermal || '—');
}

function handleSublayerEnergy(data) {
  const r = data.record;
  setEl('live-energy', (r.energy_joules || 0).toFixed(4) + ' J');
  setEl('live-latency', (r.latency_avg_ms || 0).toFixed(1) + ' ms');
  setEl('live-tps', (r.tokens_per_sec || 0).toFixed(1));
  setEl('counter-energy', data.total_records);
  addProfilingLog(
    `Energy: Block ${r.block_number}/${r.sublayer}/${r.bit_width} → ${r.energy_joules.toFixed(4)} J, ${r.tokens_per_sec.toFixed(1)} tok/s`,
    'success'
  );
}

function handleSublayerAccuracy(data) {
  const r = data.record;
  setEl('live-ppl', (r.measured_perplexity || 0).toFixed(2));
  setEl('counter-accuracy', data.total_records);
  addProfilingLog(
    `Accuracy: Block ${r.block_number}/${r.sublayer}/${r.bit_width} → PPL=${r.measured_perplexity.toFixed(2)} (Δ=${r.delta_perplexity.toFixed(2)})`,
    'success'
  );
}

function handleSublayerComplete(data) {
  addProfilingLog(
    `✓ Completed: Block ${data.block}/${data.sublayer}/${data.bit_width} (${data.step}/${data.total})`,
    'success'
  );
}

function handleProfilingCompleted(data) {
  PROF.runId = data.run_id;

  const badge = document.getElementById('prof-status-badge');
  if (badge) {
    badge.textContent = 'Completed';
    badge.className = 'badge badge-success';
  }

  const postActions = document.getElementById('prof-post-actions');
  if (postActions) postActions.classList.remove('hidden');

  // Hide pause/cancel
  hideEl('btn-live-pause');
  hideEl('btn-live-cancel');

  showToast(`Profiling complete! ${data.energy_records} energy + ${data.accuracy_records} accuracy records`, 'success');
  addProfilingLog('✓ Profiling completed successfully!', 'success');
}

function handleProfilingError(data) {
  const badge = document.getElementById('prof-status-badge');
  if (badge) {
    badge.textContent = 'Error';
    badge.className = 'badge badge-error';
  }
  showToast(data.message || 'Profiling error', 'error');
  addProfilingLog('✗ Error: ' + (data.message || 'Unknown'), 'error');
}

function handleProfilingCancelled() {
  const badge = document.getElementById('prof-status-badge');
  if (badge) {
    badge.textContent = 'Cancelled';
    badge.className = 'badge badge-warning';
  }
  showToast('Profiling cancelled', 'warning');
  addProfilingLog('Profiling cancelled by user', 'warning');
}

function handleProfilingPaused() {
  const badge = document.getElementById('prof-status-badge');
  if (badge) badge.textContent = 'Paused';
  showEl('btn-live-resume');
  hideEl('btn-live-pause');
}

function handleProfilingResumed() {
  const badge = document.getElementById('prof-status-badge');
  if (badge) badge.textContent = 'Running';
  hideEl('btn-live-resume');
  showEl('btn-live-pause');
}

// ── Configuration Page Functions ─────────────────────────────────────────

function applyResearchProfile() {
  const mode = document.getElementById('prof-research-mode')?.value;
  if (!mode || mode === 'custom') return;

  const setVal = (id, val) => { const el = document.getElementById(id); if (el) el.value = val; };
  const setCheck = (id, check) => { const el = document.getElementById(id); if (el) el.checked = check; };

  if (mode === 'quick') {
    setVal('prof-duration', '15');
    setVal('prof-dataset', 'wikitext2');
    setCheck('bw-fp16', false);
    setCheck('bw-int8', false);
    setCheck('bw-int4', true);
    setCheck('bw-int2', false);
  } else if (mode === 'research') {
    setVal('prof-duration', '30');
    setVal('prof-dataset', 'wikitext2');
    setCheck('bw-fp16', false);
    setCheck('bw-int8', true);
    setCheck('bw-int4', true);
    setCheck('bw-int2', false);
  } else if (mode === 'publication') {
    setVal('prof-duration', '60');
    setVal('prof-dataset', 'ptb');
    setCheck('bw-fp16', true);
    setCheck('bw-int8', true);
    setCheck('bw-int4', true);
    setCheck('bw-int2', true);
  }
  
  updateSchedulePreview();
}

function onProfilingModeChange() {
  const mode = document.getElementById('prof-mode')?.value;
  const sublayerGroup = document.getElementById('sublayer-group');
  if (sublayerGroup) {
    sublayerGroup.style.opacity = mode === 'single_sublayer' ? '1' : '0.5';
  }
  updateSchedulePreview();
}

function detectModelStructure(modelId) {
  fetch(`/api/profiling/structure/${encodeURIComponent(modelId)}`)
    .then(r => r.json())
    .then(data => {
      PROF.modelStructure = data;
      const preview = document.getElementById('structure-preview');
      const info = document.getElementById('structure-info');
      if (preview && info) {
        preview.style.display = 'block';
        info.innerHTML = `
          <div class="structure-detail"><strong>Model:</strong> ${data.model_id}</div>
          <div class="structure-detail"><strong>Layers:</strong> ${data.n_layers}</div>
          <div class="structure-detail"><strong>Sub-layers per block:</strong> Attention + MLP + LayerNorm</div>
          <div class="structure-detail"><strong>Total sub-layers:</strong> ${data.total_sublayers}</div>
        `;
      }
      updateSchedulePreview();
    })
    .catch(e => console.error('Structure detection failed:', e));
}

function updateSchedulePreview() {
  if (!PROF.modelStructure) return;

  const mode = document.getElementById('prof-mode')?.value || 'entire_model';
  const sublayer = document.getElementById('prof-sublayer')?.value || 'attention';
  const duration = parseInt(document.getElementById('prof-duration')?.value || '30');
  const nLayers = PROF.modelStructure.n_layers;

  // Count selected bit widths
  const bitWidths = getSelectedBitWidths();

  let totalCombinations;
  if (mode === 'single_sublayer') {
    totalCombinations = nLayers * bitWidths.length;
  } else {
    totalCombinations = nLayers * 3 * bitWidths.length;
  }

  const estimatedTime = totalCombinations * (duration + 30); // +30 for compile overhead
  const preview = document.getElementById('schedule-preview');
  const info = document.getElementById('schedule-info');

  if (preview && info) {
    preview.style.display = 'block';
    info.innerHTML = `
      <div class="structure-detail"><strong>Total combinations:</strong> ${totalCombinations}</div>
      <div class="structure-detail"><strong>Bit widths:</strong> ${bitWidths.join(', ')}</div>
      <div class="structure-detail"><strong>Estimated time:</strong> ~${formatTime(estimatedTime)}</div>
      <div class="structure-detail"><strong>Records to generate:</strong> ${totalCombinations} energy + ${totalCombinations} accuracy</div>
    `;
  }
}

function getSelectedBitWidths() {
  const bws = [];
  ['fp16', 'int8', 'int4', 'int2'].forEach(bw => {
    const cb = document.getElementById('bw-' + bw);
    if (cb && cb.checked) bws.push(bw);
  });
  return bws.length > 0 ? bws : ['int8'];
}

function startProfiling() {
  const modelId = document.getElementById('prof-model')?.value;
  if (!modelId) {
    showToast('Please select a model', 'warning');
    return;
  }

  const bitWidths = getSelectedBitWidths();
  if (bitWidths.length === 0) {
    showToast('Please select at least one bit width', 'warning');
    return;
  }

  const config = {
    model_id: modelId,
    profiling_mode: document.getElementById('prof-mode')?.value || 'entire_model',
    sublayer_type: document.getElementById('prof-sublayer')?.value || 'attention',
    bit_widths: bitWidths,
    dataset: document.getElementById('prof-dataset')?.value || 'wikitext2',
    duration_sec: parseInt(document.getElementById('prof-duration')?.value || '30'),
    warmup_runs: parseInt(document.getElementById('prof-warmup')?.value || '1'),
    max_len: parseInt(document.getElementById('prof-maxlen')?.value || '512'),
    collect_energy: document.getElementById('collect-energy')?.checked ?? true,
    collect_accuracy: document.getElementById('collect-accuracy')?.checked ?? true,
  };

  // Check if a test is already running BEFORE navigating
  fetch('/api/profiling/state')
    .then(r => r.json())
    .then(state => {
      if (state.running) {
        showToast('A profiling test is already running! Go to Live Telemetry to monitor it, or cancel it first.', 'warning');
        return;
      }
      // No test running — safe to start
      fetch('/api/profiling/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      }).then(r => r.json())
        .then(d => {
          if (d.error) {
            showToast(d.error, 'error');
          } else {
            window.location.href = '/profiling/live';
          }
        })
        .catch(e => showToast('Failed to start profiling: ' + e.message, 'error'));
    })
    .catch(() => {
      // Can't reach server, try anyway
      fetch('/api/profiling/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      }).then(r => r.json())
        .then(d => {
          if (d.error) showToast(d.error, 'error');
          else window.location.href = '/profiling/live';
        });
    });
}

function pauseProfiling() {
  fetch('/api/profiling/pause', { method: 'POST' })
    .then(r => r.json());
}

function resumeProfiling() {
  fetch('/api/profiling/resume', { method: 'POST' })
    .then(r => r.json());
}

function cancelProfiling() {
  if (typeof showConfirmDialog === 'function') {
    showConfirmDialog(
      'Cancel Profiling',
      'Are you sure? Partial data will be saved.',
      () => fetch('/api/profiling/cancel', { method: 'POST' })
    );
  } else {
    fetch('/api/profiling/cancel', { method: 'POST' });
  }
}

function deleteProfiling(runId) {
  if (typeof showConfirmDialog === 'function') {
    showConfirmDialog(
      'Delete Profiling Run',
      'This will permanently delete all data for this run.',
      () => {
        fetch(`/api/profiling/${encodeURIComponent(runId)}`, { method: 'DELETE' })
          .then(r => r.json())
          .then(() => window.location.reload());
      }
    );
  } else {
    fetch(`/api/profiling/${encodeURIComponent(runId)}`, { method: 'DELETE' })
      .then(() => window.location.reload());
  }
}

function viewResults() {
  if (PROF.runId) {
    window.location.href = '/profiling/results/' + encodeURIComponent(PROF.runId);
  }
}

function exportResults(fmt) {
  if (PROF.runId) {
    window.location.href = `/api/profiling/${encodeURIComponent(PROF.runId)}/export?format=${fmt}`;
  }
}

// ── Live Chart ───────────────────────────────────────────────────────────

function initProfilingLiveChart() {
  const canvas = document.getElementById('prof-live-chart');
  if (!canvas || typeof Chart === 'undefined') return;

  PROF.liveChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'CPU (mW)', data: [],
          borderColor: '#60a5fa', backgroundColor: 'rgba(96,165,250,0.1)',
          borderWidth: 2, fill: true, tension: 0.3, pointRadius: 0,
        },
        {
          label: 'ANE (mW)', data: [],
          borderColor: '#a78bfa', backgroundColor: 'rgba(167,139,250,0.1)',
          borderWidth: 2, fill: true, tension: 0.3, pointRadius: 0,
        },
        {
          label: 'GPU (mW)', data: [],
          borderColor: '#34d399', backgroundColor: 'rgba(52,211,153,0.1)',
          borderWidth: 2, fill: true, tension: 0.3, pointRadius: 0,
        },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#5e5e78', maxTicksLimit: 10 } },
        y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#5e5e78' }, beginAtZero: true },
      },
      plugins: {
        legend: { labels: { color: '#9898b0', usePointStyle: true, pointStyle: 'circle' } },
      }
    }
  });
}

// ── Heatmaps ─────────────────────────────────────────────────────────────

function renderEnergyHeatmap(records) {
  const container = document.getElementById('energy-heatmap');
  if (!container) return;
  renderHeatmap(container, records, 'energy_joules', 'Energy (J)', 'warm');
}

function renderAccuracyHeatmap(records) {
  const container = document.getElementById('accuracy-heatmap');
  if (!container) return;
  renderHeatmap(container, records, 'delta_perplexity', 'ΔPPL', 'cool');
}

function renderHeatmap(container, records, valueKey, label, colorScheme) {
  // Group by sublayer+bitwidth vs block
  const blocks = [...new Set(records.map(r => r.block_number))].sort((a, b) => a - b);
  const keys = [...new Set(records.map(r => r.sublayer + '/' + r.bit_width))];

  // Find value range
  const values = records.map(r => r[valueKey] || 0);
  const vMin = Math.min(...values);
  const vMax = Math.max(...values);
  const range = vMax - vMin || 1;

  let html = '<table class="heatmap-table"><thead><tr><th></th>';
  blocks.forEach(b => html += `<th>B${b}</th>`);
  html += '</tr></thead><tbody>';

  keys.forEach(key => {
    html += `<tr><td class="heatmap-row-label">${key}</td>`;
    blocks.forEach(block => {
      const record = records.find(r =>
        r.block_number === block && (r.sublayer + '/' + r.bit_width) === key
      );
      const val = record ? (record[valueKey] || 0) : 0;
      const intensity = (val - vMin) / range;
      const color = colorScheme === 'warm'
        ? `rgba(248,113,113,${0.1 + intensity * 0.8})`
        : `rgba(96,165,250,${0.1 + intensity * 0.8})`;
      html += `<td class="heatmap-cell" style="background:${color}" title="${key} Block ${block}: ${val.toFixed(4)}">${val.toFixed(3)}</td>`;
    });
    html += '</tr>';
  });

  html += '</tbody></table>';
  container.innerHTML = html;
}

// ── Charts ───────────────────────────────────────────────────────────────

function renderScatterChart(energyRecords, accuracyRecords) {
  const canvas = document.getElementById('scatter-chart');
  if (!canvas || typeof Chart === 'undefined') return;

  const points = [];
  energyRecords.forEach(er => {
    const ar = accuracyRecords.find(a =>
      a.block_number === er.block_number &&
      a.sublayer === er.sublayer &&
      a.bit_width === er.bit_width
    );
    if (ar) {
      points.push({
        x: er.energy_joules,
        y: ar.delta_perplexity,
        label: `B${er.block_number}/${er.sublayer}/${er.bit_width}`,
      });
    }
  });

  const colors = { attention: '#60a5fa', mlp: '#34d399', layernorm: '#fbbf24' };

  new Chart(canvas, {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'Sub-layers',
        data: points.map(p => ({ x: p.x, y: p.y })),
        backgroundColor: points.map(p => {
          const sl = p.label.split('/')[1];
          return colors[sl] || '#a78bfa';
        }),
        pointRadius: 6,
        pointHoverRadius: 8,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: 'Energy (J)', color: '#9898b0' },
             grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#5e5e78' } },
        y: { title: { display: true, text: 'ΔPPL', color: '#9898b0' },
             grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#5e5e78' } },
      },
      plugins: { legend: { display: false } },
    }
  });
}

function renderSensitivityChart(records) {
  const canvas = document.getElementById('sensitivity-chart');
  if (!canvas || typeof Chart === 'undefined') return;

  // Sort by delta_perplexity (highest = most sensitive)
  const sorted = [...records].sort((a, b) => (b.delta_perplexity || 0) - (a.delta_perplexity || 0)).slice(0, 20);

  new Chart(canvas, {
    type: 'bar',
    data: {
      labels: sorted.map(r => `B${r.block_number}/${r.sublayer}/${r.bit_width}`),
      datasets: [{
        label: 'ΔPPL',
        data: sorted.map(r => r.delta_perplexity || 0),
        backgroundColor: sorted.map(r => {
          const val = r.delta_perplexity || 0;
          return val > 1 ? 'rgba(248,113,113,0.7)' : val > 0.1 ? 'rgba(251,191,36,0.7)' : 'rgba(52,211,153,0.7)';
        }),
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, indexAxis: 'y',
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#5e5e78' },
             title: { display: true, text: 'ΔPPL', color: '#9898b0' } },
        y: { grid: { display: false }, ticks: { color: '#9898b0', font: { size: 10 } } },
      },
      plugins: { legend: { display: false } },
    }
  });
}

function renderEnergyRanking(records) {
  const canvas = document.getElementById('energy-ranking-chart');
  if (!canvas || typeof Chart === 'undefined') return;

  const sorted = [...records].sort((a, b) => (b.energy_joules || 0) - (a.energy_joules || 0)).slice(0, 20);

  new Chart(canvas, {
    type: 'bar',
    data: {
      labels: sorted.map(r => `B${r.block_number}/${r.sublayer}/${r.bit_width}`),
      datasets: [{
        label: 'Energy (J)',
        data: sorted.map(r => r.energy_joules || 0),
        backgroundColor: 'rgba(167,139,250,0.6)',
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, indexAxis: 'y',
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#5e5e78' } },
        y: { grid: { display: false }, ticks: { color: '#9898b0', font: { size: 10 } } },
      },
      plugins: { legend: { display: false } },
    }
  });
}

function renderLatencyChart(records) {
  const canvas = document.getElementById('latency-chart');
  if (!canvas || typeof Chart === 'undefined') return;

  const sorted = [...records].sort((a, b) => (b.latency_avg_ms || 0) - (a.latency_avg_ms || 0)).slice(0, 20);

  new Chart(canvas, {
    type: 'bar',
    data: {
      labels: sorted.map(r => `B${r.block_number}/${r.sublayer}/${r.bit_width}`),
      datasets: [{
        label: 'Latency (ms)',
        data: sorted.map(r => r.latency_avg_ms || 0),
        backgroundColor: 'rgba(96,165,250,0.6)',
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, indexAxis: 'y',
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#5e5e78' } },
        y: { grid: { display: false }, ticks: { color: '#9898b0', font: { size: 10 } } },
      },
      plugins: { legend: { display: false } },
    }
  });
}

// ── Utility Functions ────────────────────────────────────────────────────

function setEl(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function showEl(id) {
  const el = document.getElementById(id);
  if (el) el.classList.remove('hidden');
}

function hideEl(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add('hidden');
}

function formatTime(seconds) {
  seconds = Math.round(seconds);
  if (seconds < 60) return seconds + 's';
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m < 60) return m + 'm ' + s + 's';
  const h = Math.floor(m / 60);
  return h + 'h ' + (m % 60) + 'm';
}

function addProfilingLog(message, type = '') {
  const console_el = document.getElementById('prof-console');
  if (!console_el) return;

  const entry = document.createElement('div');
  entry.className = 'log-entry ' + type;
  const time = new Date().toLocaleTimeString();
  entry.textContent = `[${time}] ${message}`;
  console_el.appendChild(entry);
  console_el.scrollTop = console_el.scrollHeight;
}

function clearConsole() {
  const console_el = document.getElementById('prof-console');
  if (console_el) console_el.innerHTML = '';
}

function restoreProfilingState() {
  // Only poll on the live page (check for the live progress bar)
  if (!document.getElementById('prof-progress-bar')) return;
  
  fetch('/api/profiling/state')
    .then(r => r.json())
    .then(data => {
      if (data.running) {
        // Update UI with the latest server-side state (fromPoll=true to skip console spam)
        handleProfilingProgress(data, true);
        if (data.state === 'paused') {
          handleProfilingPaused();
        } else {
          handleProfilingResumed();
        }
        
        // Restore live metric cards from cached values
        if (data.last_metric) {
          const m = data.last_metric;
          setEl('live-power', (m.compute_mw || 0).toFixed(1) + ' mW');
          setEl('live-temp', m.thermal || '—');
        }
        if (data.last_energy) {
          const r = data.last_energy;
          if (r.energy_joules !== undefined) setEl('live-energy', r.energy_joules.toFixed(4) + ' J');
          if (r.latency_avg_ms !== undefined) setEl('live-latency', r.latency_avg_ms.toFixed(1) + ' ms');
          if (r.tokens_per_sec !== undefined) setEl('live-tps', r.tokens_per_sec.toFixed(1));
        }
        if (data.last_accuracy) {
          const r = data.last_accuracy;
          if (r.measured_perplexity !== undefined) setEl('live-ppl', r.measured_perplexity.toFixed(2));
        }
      } else {
        // Test not running — if it was before, it finished
        // Stop polling
        if (PROF._statePoller) {
          clearInterval(PROF._statePoller);
          PROF._statePoller = null;
        }
      }
    })
    .catch(() => {});
}

// ── Init ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initProfilingSocket();
  // Restore state immediately and then poll every 2 seconds
  // This ensures the live page always shows fresh data even
  // if WebSocket events were missed during a page navigation
  restoreProfilingState();
  PROF._statePoller = setInterval(restoreProfilingState, 2000);
});
