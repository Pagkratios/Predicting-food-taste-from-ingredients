/* ═══════════════════════════════════════════════════════════════════════════
   FlavorLab – main.js
   Handles tab switching, file upload, form submission, and result rendering
═══════════════════════════════════════════════════════════════════════════ */

// ── Sense config ─────────────────────────────────────────────────────────────
const SENSES = ['sweet', 'bitter', 'salty', 'umami', 'sour'];
const SENSE_COLORS = {
  sweet:  '#4477AA',
  bitter: '#EE6677',
  salty:  '#228833',
  umami:  '#CCBB44',
  sour:   '#AA3377',
};
const SENSE_LABELS = {
  sweet: 'Sweet', bitter: 'Bitter', salty: 'Salty', umami: 'Umami', sour: 'Sour',
};

// ── DOM refs ─────────────────────────────────────────────────────────────────
const tabs        = document.querySelectorAll('.tab-btn');
const panes       = document.querySelectorAll('.tab-pane');
const textarea    = document.getElementById('recipeText');
const fileInput   = document.getElementById('fileInput');
const dropZone    = document.getElementById('dropZone');
const fileBadge   = document.getElementById('fileBadge');
const fileBadgeName = document.getElementById('fileBadgeName');
const predictBtn  = document.getElementById('predictBtn');
const loadingEl   = document.getElementById('loadingOverlay');
const errorEl     = document.getElementById('errorBanner');
const errorMsg    = document.getElementById('errorMsg');
const resultsEl   = document.getElementById('resultsSection');

let radarChart    = null;

// ── Tab switching ─────────────────────────────────────────────────────────────
tabs.forEach(btn => {
  btn.addEventListener('click', () => {
    tabs.forEach(t  => t.classList.remove('active'));
    panes.forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`pane-${btn.dataset.tab}`).classList.add('active');
  });
});

// ── File input handling ───────────────────────────────────────────────────────
function showFileBadge(name) {
  fileBadgeName.textContent = name;
  fileBadge.style.display = 'inline-flex';
}

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) showFileBadge(fileInput.files[0].name);
});

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) {
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    showFileBadge(file.name);
  }
});

// ── Loading steps animation ──────────────────────────────────────────────────
const stepEls = document.querySelectorAll('.loading-steps li');
let stepTimer = null;

function animateSteps() {
  let idx = 0;
  stepEls.forEach(el => el.classList.remove('active', 'done'));
  function advance() {
    if (idx > 0) stepEls[idx - 1].classList.replace('active', 'done');
    if (idx < stepEls.length) {
      stepEls[idx].classList.add('active');
      idx++;
      stepTimer = setTimeout(advance, 2200);
    }
  }
  advance();
}
function stopSteps() {
  clearTimeout(stepTimer);
  stepEls.forEach(el => el.classList.remove('active'));
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function showError(msg) {
  errorMsg.textContent = msg;
  errorEl.classList.add('visible');
  loadingEl.classList.remove('visible');
  stopSteps();
  predictBtn.disabled = false;
}

function hideError() { errorEl.classList.remove('visible'); }

function pct(val) { return `${Math.round(val * 100)}%`; }

// ── Radar chart ───────────────────────────────────────────────────────────────
function buildRadar(predictions, confidence) {
  const ctx = document.getElementById('radarCanvas').getContext('2d');
  if (radarChart) radarChart.destroy();

  const data   = SENSES.map(s => predictions[s] ?? 0);
  const lower  = SENSES.map(s => confidence[s]?.lower ?? 0);
  const upper  = SENSES.map(s => confidence[s]?.upper ?? 0);
  const labels = SENSES.map(s => SENSE_LABELS[s]);

  // Build gradient fill
  const gradient = ctx.createRadialGradient(
    ctx.canvas.width / 2, ctx.canvas.height / 2, 0,
    ctx.canvas.width / 2, ctx.canvas.height / 2, ctx.canvas.width / 2
  );
  gradient.addColorStop(0,   'rgba(196, 151, 80, 0.25)');
  gradient.addColorStop(1,   'rgba(196, 151, 80, 0.02)');

  radarChart = new Chart(ctx, {
    type: 'radar',
    data: {
      labels,
      datasets: [
        {
          label: 'Confidence range',
          data: upper,
          borderColor: 'transparent',
          backgroundColor: 'rgba(196,151,80,0.08)',
          pointRadius: 0,
          fill: true,
        },
        {
          label: 'Predicted',
          data,
          borderColor: '#D4A843',
          borderWidth: 2,
          backgroundColor: gradient,
          pointBackgroundColor: SENSES.map(s => SENSE_COLORS[s]),
          pointBorderColor: '#111827',
          pointRadius: 5,
          pointHoverRadius: 7,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      animation: { duration: 900, easing: 'easeOutQuart' },
      scales: {
        r: {
          min: 0,
          max: 100,
          ticks: {
            stepSize: 25,
            color: '#4A506A',
            font: { size: 9, family: "'Plus Jakarta Sans'" },
            backdropColor: 'transparent',
          },
          grid: {
            color: 'rgba(255,255,255,0.05)',
            lineWidth: 1,
          },
          angleLines: { color: 'rgba(255,255,255,0.06)', lineWidth: 1 },
          pointLabels: {
            font: { size: 12, family: "'Cormorant Garamond'", weight: '600' },
            color: labels.map((_, i) => SENSE_COLORS[SENSES[i]]),
          },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#111827',
          borderColor: 'rgba(196,151,80,0.3)',
          borderWidth: 1,
          titleColor: '#E8C97A',
          bodyColor: '#EDE8DD',
          padding: 10,
          callbacks: {
            label: ctx => {
              const s = SENSES[ctx.dataIndex];
              if (ctx.datasetIndex === 1) {
                const lo = confidence[s]?.lower ?? 0;
                const hi = confidence[s]?.upper ?? 0;
                return ` ${ctx.parsed.r.toFixed(1)}  (${lo.toFixed(1)}–${hi.toFixed(1)})`;
              }
              return '';
            },
          },
        },
      },
    },
  });
}

// ── Recipe table ──────────────────────────────────────────────────────────────
function buildRecipeTable(recipe) {
  document.getElementById('recipeName').textContent = recipe.recipe_name;

  const tbody = document.getElementById('ingredientTbody');
  tbody.innerHTML = '';

  recipe.ingredients.forEach(ing => {
    const w   = ing.weight;
    const s   = ing.sensory_scores;
    const row = document.createElement('tr');

    // Name
    const tdName = document.createElement('td');
    tdName.innerHTML = `<span class="ingredient-name">${ing.name}</span>`;

    // Weight bar
    const tdWeight = document.createElement('td');
    tdWeight.className = 'weight-bar-cell';
    tdWeight.innerHTML = `
      <div class="weight-bar-wrap">
        <div class="weight-bar-bg">
          <div class="weight-bar" style="width:${Math.min(w * 100, 100)}%"></div>
        </div>
        <span class="weight-pct">${(w * 100).toFixed(1)}%</span>
      </div>`;

    // Sensory chips
    const tdSensory = document.createElement('td');
    tdSensory.innerHTML = SENSES.map(sense => {
      const v = s[sense] ?? '–';
      return `<span class="sense-chip" style="background:${SENSE_COLORS[sense]}22;color:${SENSE_COLORS[sense]};border:1px solid ${SENSE_COLORS[sense]}44"><span class="chip-label">${SENSE_LABELS[sense]}</span><span class="chip-val">${v}</span></span>`;
    }).join('');

    row.append(tdName, tdWeight, tdSensory);
    tbody.appendChild(row);
  });
}

// ── Scores table ──────────────────────────────────────────────────────────────
function buildScoresTable(predictions, confidence) {
  const container = document.getElementById('scoresContainer');
  container.innerHTML = '';

  SENSES.forEach(sense => {
    const pred  = predictions[sense] ?? 0;
    const lo    = confidence[sense]?.lower ?? 0;
    const hi    = confidence[sense]?.upper ?? 0;
    const color = SENSE_COLORS[sense];

    const row = document.createElement('div');
    row.className = 'score-row';
    row.innerHTML = `
      <div class="score-sense-name">
        <div class="sense-indicator" style="background:${color}"></div>
        ${SENSE_LABELS[sense]}
      </div>
      <div class="score-gauge-wrap">
        <div class="score-gauge-bg">
          <div class="score-gauge-range" style="left:${lo}%;width:${hi - lo}%;background:${color}"></div>
          <div class="score-gauge-fill" style="width:${pred}%;background:${color}"></div>
        </div>
      </div>
      <div class="score-values">
        <div class="score-main" style="color:${color}">${pred.toFixed(1)}</div>
        <div class="score-range">${lo.toFixed(1)} – ${hi.toFixed(1)}</div>
      </div>`;

    container.appendChild(row);
  });
}

// ── Main submit ───────────────────────────────────────────────────────────────
predictBtn.addEventListener('click', async () => {
  hideError();

  // Determine active tab
  const activeTab = document.querySelector('.tab-btn.active').dataset.tab;

  const formData = new FormData();
  if (activeTab === 'text') {
    const txt = textarea.value.trim();
    if (!txt) { showError('Please paste a recipe description first.'); return; }
    formData.append('text', txt);
  } else {
    if (!fileInput.files[0]) { showError('Please select a file to upload.'); return; }
    formData.append('file', fileInput.files[0]);
  }

  // Show loading
  predictBtn.disabled = true;
  resultsEl.classList.remove('visible');
  loadingEl.classList.add('visible');
  animateSteps();

  try {
    const res  = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();

    stopSteps();
    loadingEl.classList.remove('visible');
    predictBtn.disabled = false;

    if (!res.ok || data.error) {
      if (data.missing_scores) {
        showError(data.error);
      } else {
        showError(data.error ?? `Server error: ${res.status}`);
      }
      return;
    }

    const { recipe, predictions, confidence } = data;

    buildRecipeTable(recipe);
    buildScoresTable(predictions, confidence);
    buildRadar(predictions, confidence);

    document.getElementById('estimateBanner').style.display = 'none';
    resultsEl.classList.add('visible');
    resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });

  } catch (err) {
    showError(`Network error: ${err.message}`);
  }
});

// ── Allow Enter in textarea to keep newlines (block form submit) ─────────────
textarea.addEventListener('keydown', e => {
  if (e.key === 'Enter' && e.metaKey) predictBtn.click();
});
