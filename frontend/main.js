/* ============================================================
   main.js — EEG Neural Decode Dashboard
   ============================================================ */

'use strict';

const API = '';
let waveformChart  = null;
let distChart      = null;
let currentEpoch   = 0;
let currentModel   = 'eegnet';
let maxEpochIdx    = 134;
let debounceTimer  = null;

// Colour palette for 8 waveform channels (green -> cyan gradient)
const WAVE_COLORS = [
  '#00ff88','#00e87a','#00d46e','#00cfff',
  '#44ffaa','#22ddcc','#00bbdd','#66eebb',
];

// ============================================================
//  Boot
// ============================================================
(async function init() {
  startBrainAnimation();
  await checkStatus();
  await loadOverview();
  await loadEpoch(0);
  await loadComparison();
  setupControls();
})();

// ============================================================
//  [A] Brain canvas animation
// ============================================================
function startBrainAnimation() {
  const canvas = document.getElementById('brain-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function resize() {
    canvas.width  = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  // ~90 dots arranged in an oval (brain shape)
  const dots = [];
  for (let i = 0; i < 90; i++) {
    const angle = Math.random() * Math.PI * 2;
    const r     = 0.55 + Math.random() * 0.35;
    dots.push({
      nx:            0.5 + Math.cos(angle) * 0.38 * r,
      ny:            0.5 + Math.sin(angle) * 0.32 * r,
      phase:         Math.random() * Math.PI * 2,
      speed:         0.4 + Math.random() * 0.8,
      size:          1.2 + Math.random() * 1.6,
      flashTimer:    0,
      flashDuration: 0.3 + Math.random() * 0.35,
    });
  }

  // Schedule random neural "fire" flashes
  ;(function scheduleFire() {
    const d = dots[Math.floor(Math.random() * dots.length)];
    d.flashTimer = 1.0;
    setTimeout(scheduleFire, 90 + Math.random() * 300);
  })();

  let last = performance.now();
  function draw(now) {
    const dt = (now - last) / 1000;
    last = now;
    canvas.width  = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    const W = canvas.width, H = canvas.height;

    for (const d of dots) {
      const base  = 0.15 + 0.1 * Math.sin(now * 0.001 * d.speed + d.phase);
      if (d.flashTimer > 0) d.flashTimer = Math.max(0, d.flashTimer - dt / d.flashDuration);
      const alpha = Math.min(1, base + d.flashTimer * 0.72);
      const s     = d.size + d.flashTimer * 2.2;

      ctx.beginPath();
      ctx.arc(d.nx * W, d.ny * H, s, 0, Math.PI * 2);
      if (d.flashTimer > 0.25) {
        ctx.fillStyle  = `rgba(0,207,255,${alpha})`;
        ctx.shadowBlur = 8;
        ctx.shadowColor = 'rgba(0,207,255,0.5)';
      } else {
        ctx.fillStyle  = `rgba(0,255,136,${alpha})`;
        ctx.shadowBlur = 4;
        ctx.shadowColor = 'rgba(0,255,136,0.35)';
      }
      ctx.fill();
      ctx.shadowBlur = 0;
    }
    requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
}

// ============================================================
//  API helpers
// ============================================================
async function apiJSON(path) {
  try {
    const res = await fetch(API + path);
    if (!res.ok) {
      console.warn(`API ${path} -> HTTP ${res.status}`);
      return null;
    }
    return res.json();
  } catch (e) {
    console.warn('API fetch error:', path, e);
    return null;
  }
}

// ============================================================
//  Status check
// ============================================================
async function checkStatus() {
  const data = await apiJSON('/api/status');
  if (!data) return;

  if (!data.data_loaded) {
    const banner = document.getElementById('warning-banner');
    if (banner) banner.style.display = 'block';
  }
  if (data.n_epochs > 0) {
    maxEpochIdx = data.n_epochs - 1;
    const slider = document.getElementById('epoch-slider');
    if (slider) slider.max = maxEpochIdx;
    const maxEl = document.getElementById('epoch-max');
    if (maxEl) maxEl.textContent = maxEpochIdx;
  }
}

// ============================================================
//  Overview
// ============================================================
async function loadOverview() {
  const data = await apiJSON('/api/overview');
  if (!data) return;

  setText('m-epochs',   data.n_epochs);
  setText('m-channels', data.n_channels);
  setText('m-times',    data.n_times);
  setText('m-duration', data.duration_s + 's');

  maxEpochIdx = data.n_epochs - 1;
  const slider = document.getElementById('epoch-slider');
  if (slider) slider.max = maxEpochIdx;
  const maxEl = document.getElementById('epoch-max');
  if (maxEl) maxEl.textContent = maxEpochIdx;

  buildDistChart(data.class_counts);
}

function buildDistChart(counts) {
  const labels = Object.keys(counts);
  const values = Object.values(counts);
  const canvas = document.getElementById('dist-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  if (distChart) distChart.destroy();
  distChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: ['rgba(0,255,136,0.55)', 'rgba(0,207,255,0.55)'],
        borderColor:     ['#00ff88', '#00cfff'],
        borderWidth: 1,
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          grid:   { color: '#1a1a1a' },
          ticks:  { color: '#888', font: { family: "'JetBrains Mono', monospace", size: 11 } },
          border: { color: '#2a2a2a' },
        },
        y: {
          grid:   { display: false },
          ticks:  { color: '#e0e0e0', font: { family: "'JetBrains Mono', monospace", size: 12 } },
          border: { color: '#2a2a2a' },
        },
      },
    },
  });
}

// ============================================================
//  Epoch playback
// ============================================================
async function loadEpoch(idx) {
  const data = await apiJSON(`/api/epoch/${idx}?model=${currentModel}`);
  if (!data) return;

  currentEpoch = idx;

  // Labels
  setText('true-label', data.true_label);
  const predEl = document.getElementById('pred-label');
  if (predEl) {
    predEl.textContent = data.predicted_label;
    predEl.className   = 'pred-value ' + (data.correct ? 'correct' : 'incorrect');
  }

  // Confidence bars (rAF ensures transition fires after layout)
  const pLeft  = data.probabilities['Left Hand']  || 0;
  const pRight = data.probabilities['Right Hand'] || 0;
  setText('prob-left',  (pLeft  * 100).toFixed(1) + '%');
  setText('prob-right', (pRight * 100).toFixed(1) + '%');
  requestAnimationFrame(() => {
    setStyle('bar-left',  'width', (pLeft  * 100) + '%');
    setStyle('bar-right', 'width', (pRight * 100) + '%');
  });

  // Meta line
  setText('epoch-meta',
    `model: ${data.model}  |  epoch: ${idx}  |  channels shown: ${data.waveform.length}`);

  buildWaveform(data.time_axis, data.waveform, data.channel_names);

  // GradCAM: only fetch for EEGNet, show info panel otherwise
  updateGradCAM(idx);
}

// ============================================================
//  Waveform
// ============================================================
function buildWaveform(timeAxis, waveform, chNames) {
  const canvas = document.getElementById('waveform-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const datasets = waveform.map((ch, i) => {
    const offset = (waveform.length - 1 - i) * 3;
    return {
      label:       chNames ? chNames[i] : `Ch${i}`,
      data:        ch.map((v, j) => ({ x: timeAxis[j], y: v + offset })),
      borderColor: WAVE_COLORS[i % WAVE_COLORS.length],
      borderWidth: 1,
      pointRadius: 0,
      tension:     0,
    };
  });

  // Vertical line at t=0 (event onset)
  const yMax = waveform.length * 3 + 2;
  const onsetLine = {
    label:       '_onset',
    data:        [{ x: 0, y: -2 }, { x: 0, y: yMax }],
    borderColor: 'rgba(255,68,102,0.65)',
    borderWidth: 1.5,
    borderDash:  [4, 4],
    pointRadius: 0,
    tension:     0,
    order:       -1,
  };

  if (waveformChart) {
    waveformChart.data.datasets = [...datasets, onsetLine];
    waveformChart.update('active');
    return;
  }

  waveformChart = new Chart(ctx, {
    type: 'line',
    data: { datasets: [...datasets, onsetLine] },
    options: {
      responsive:         true,
      maintainAspectRatio: false,
      animation:          { duration: 250 },
      parsing:            false,
      plugins: {
        legend: { display: false },
        tooltip: {
          filter: item => item.dataset.label !== '_onset',
          callbacks: {
            title: items => `t = ${(+items[0].raw.x).toFixed(3)} s`,
            label: item  => ` ${item.dataset.label}: ${(+item.raw.y).toFixed(2)}`,
          },
        },
      },
      scales: {
        x: {
          type:   'linear',
          title:  { display: true, text: 'Time (s)', color: '#666',
                    font: { family: "'JetBrains Mono', monospace", size: 10 } },
          grid:   { color: '#1a1a1a' },
          ticks:  { color: '#666',
                    font: { family: "'JetBrains Mono', monospace", size: 10 },
                    maxTicksLimit: 8 },
          border: { color: '#2a2a2a' },
        },
        y: {
          display: false,
          grid:    { display: false },
        },
      },
    },
  });
}

// ============================================================
//  GradCAM
// ============================================================
function updateGradCAM(idx) {
  const imgEl     = document.getElementById('gradcam-img');
  const topChEl   = document.getElementById('top-channels');
  const panelCard = document.getElementById('gradcam-panel');

  if (!imgEl) return;

  // GradCAM is only meaningful for EEGNet
  if (currentModel !== 'eegnet') {
    imgEl.src = '';
    imgEl.style.opacity = '0.3';
    if (topChEl) topChEl.innerHTML =
      `&gt; <span class="muted">GradCAM is available for EEGNet only.</span>` +
      ` <span class="muted">Switch model to EEGNet to see the topomap.</span>`;
    return;
  }

  imgEl.style.opacity   = '0.4';
  imgEl.style.filter    = 'blur(2px)';
  if (topChEl) topChEl.innerHTML = `&gt; <span class="muted">computing...</span>`;

  // Single fetch: use response blob as img src + read header
  const url = `/api/gradcam/${idx}`;
  fetch(url)
    .then(res => {
      const topCh = res.headers.get('X-Top-Channels');
      if (topChEl) {
        if (topCh && topCh !== 'N/A') {
          topChEl.innerHTML =
            `&gt; top activated electrodes: <span class="cyan">${topCh}</span>`;
        } else {
          topChEl.innerHTML = `&gt; <span class="muted">electrode data unavailable</span>`;
        }
      }
      return res.blob();
    })
    .then(blob => {
      const objectURL = URL.createObjectURL(blob);
      imgEl.onload = () => {
        imgEl.style.opacity = '1';
        imgEl.style.filter  = '';
        // Release old blob URL after image loads
        if (imgEl.dataset.blobUrl) URL.revokeObjectURL(imgEl.dataset.blobUrl);
        imgEl.dataset.blobUrl = objectURL;
      };
      imgEl.src = objectURL;
    })
    .catch(e => {
      console.warn('GradCAM fetch error:', e);
      imgEl.style.opacity = '1';
      imgEl.style.filter  = '';
      if (topChEl) topChEl.innerHTML = `&gt; <span class="red">GradCAM error</span>`;
    });
}

// ============================================================
//  Model comparison table
// ============================================================
async function loadComparison() {
  const data = await apiJSON('/api/comparison');
  if (!data) return;

  const modelMeta = {
    eegnet:   'EEGNet',
    cnn_lstm: 'CNN-LSTM',
    mdm:      'Riemannian TS+LR',
  };

  // Find best accuracy for highlight
  const accs    = Object.values(data).map(r => r?.accuracy ?? 0);
  const bestAcc = Math.max(...accs);

  const tbody = document.getElementById('bench-body');
  if (!tbody) return;
  tbody.innerHTML = '';

  for (const [key, r] of Object.entries(data)) {
    if (!r || r.accuracy == null) continue;
    const isBest = r.accuracy === bestAcc;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="mono">${modelMeta[key] || key}</td>
      <td class="mono ${isBest ? 'acc-best' : ''}">${(r.accuracy * 100).toFixed(1)}%</td>
      <td class="mono">${r.kappa != null ? r.kappa.toFixed(4) : '&mdash;'}</td>
      <td><span class="type-badge">${r.type || ''}</span></td>
    `;
    tbody.appendChild(tr);
  }

  // Gracefully hide missing training curve images
  ['eegnet', 'cnnlstm'].forEach(m => {
    const img = document.getElementById(`curve-${m}`);
    if (!img) return;
    img.onerror = () => {
      const wrap = img.parentElement;
      img.remove();
      const p = document.createElement('p');
      p.className = 'muted';
      p.style.fontSize = '12px';
      p.style.marginTop = '8px';
      p.textContent = 'Training curve not found.';
      wrap.appendChild(p);
    };
  });
}

// ============================================================
//  Controls
// ============================================================
function setupControls() {
  const slider     = document.getElementById('epoch-slider');
  const epochDisp  = document.getElementById('epoch-display');
  const modelSelEl = document.getElementById('model-sel');

  if (slider) {
    slider.addEventListener('input', () => {
      const idx = parseInt(slider.value, 10);
      if (epochDisp) epochDisp.textContent = idx;
      clearTimeout(debounceTimer);
      // 200 ms debounce — prevents API flood on fast slider drag
      debounceTimer = setTimeout(() => loadEpoch(idx), 200);
    });
  }

  if (modelSelEl) {
    modelSelEl.addEventListener('change', () => {
      currentModel = modelSelEl.value;
      // Cancel any pending debounced epoch fetch and load immediately
      clearTimeout(debounceTimer);
      loadEpoch(currentEpoch);
    });
  }

  // Subject selector is cosmetic (all test subjects' epochs are concatenated)
  const subjectSel = document.getElementById('subject-sel');
  if (subjectSel) {
    subjectSel.addEventListener('change', () => {
      if (slider)    { slider.value = 0; }
      if (epochDisp) { epochDisp.textContent = 0; }
      loadEpoch(0);
    });
  }
}

// ============================================================
//  Tiny DOM helpers
// ============================================================
function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}
function setStyle(id, prop, value) {
  const el = document.getElementById(id);
  if (el) el.style[prop] = value;
}
