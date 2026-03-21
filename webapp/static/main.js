/* FlavorLab – main.js */

const SENSES = ['sweet', 'bitter', 'salty', 'umami', 'sour'];
const SENSE_COLORS  = { sweet:'#4477AA', bitter:'#EE6677', salty:'#228833', umami:'#CCBB44', sour:'#AA3377' };
const SENSE_LABELS  = { sweet:'Sweet', bitter:'Bitter', salty:'Salty', umami:'Umami', sour:'Sour' };

// DOM refs
const mainInput      = document.getElementById('mainInput');
const submitBtn      = document.getElementById('submitBtn');
const attachBtn      = document.getElementById('attachBtn');
const fileInput      = document.getElementById('fileInput');
const fileChip       = document.getElementById('fileChip');
const fileChipName   = document.getElementById('fileChipName');
const fileChipRemove = document.getElementById('fileChipRemove');
const chatInputBox   = document.getElementById('chatInputBox');
const centerView     = document.getElementById('centerView');
const heroContent    = document.getElementById('heroContent');
const stepsPanel     = document.getElementById('stepsPanel');
const stepListEl     = document.getElementById('stepList');
const resultsLayout  = document.getElementById('resultsLayout');
const errorBanner    = document.getElementById('errorBanner');
const errorMsg       = document.getElementById('errorMsg');

let radarChart  = null;
let stepTimer   = null;
let currentFile = null;

// ── State machine ──────────────────────────────────────────────────────────
// All display toggling is done via inline styles (avoids CSS specificity fights).
// CSS classes on <body> drive layout/positioning only (not display).

function setState(state) {
  document.body.className = `state-${state}`;

  if (state === 'initial') {
    centerView.style.display    = '';
    centerView.style.opacity    = '1';
    centerView.style.transition = '';
    heroContent.style.display   = '';
    heroContent.style.opacity   = '1';
    heroContent.style.transform = '';
    heroContent.style.transition = '';
    stepsPanel.style.display    = 'none';
    resultsLayout.style.display = 'none';
    resultsLayout.style.opacity = '';
    submitBtn.disabled = false;
  }

  if (state === 'loading') {
    // Fade + slide hero out
    heroContent.style.transition = 'opacity 0.25s ease, transform 0.3s ease';
    heroContent.style.opacity    = '0';
    heroContent.style.transform  = 'translateY(-14px)';
    setTimeout(() => {
      if (document.body.classList.contains('state-loading')) {
        heroContent.style.display = 'none';
      }
    }, 310);

    centerView.style.display    = '';
    centerView.style.opacity    = '1';
    stepsPanel.style.display    = 'block';
    stepsPanel.style.animation  = 'none'; // reset then re-trigger
    requestAnimationFrame(() => {
      stepsPanel.style.animation = '';
    });
    resultsLayout.style.display = 'none';
    submitBtn.disabled = true;
  }

  if (state === 'results') {
    // Fade center-view out
    centerView.style.transition = 'opacity 0.25s ease';
    centerView.style.opacity    = '0';
    setTimeout(() => {
      if (document.body.classList.contains('state-results')) {
        centerView.style.display = 'none';
      }
    }, 270);

    // Fade results in
    resultsLayout.style.display    = 'grid';
    resultsLayout.style.opacity    = '0';
    resultsLayout.style.transition = '';
    requestAnimationFrame(() => {
      resultsLayout.style.transition = 'opacity 0.4s ease';
      resultsLayout.style.opacity    = '1';
    });
    submitBtn.disabled = false;
  }
}

// ── Step animation ─────────────────────────────────────────────────────────
const stepEls = Array.from(stepListEl.querySelectorAll('li'));

function startSteps() {
  let i = 0;
  stepEls.forEach(el => el.className = '');
  function tick() {
    if (i > 0) stepEls[i - 1].classList.add('done');
    if (i < stepEls.length) {
      stepEls[i].classList.add('active');
      i++;
      stepTimer = setTimeout(tick, 2200);
    }
  }
  tick();
}
function stopSteps() {
  clearTimeout(stepTimer);
  stepEls.forEach(el => el.classList.remove('active'));
  stepEls.forEach(el => el.classList.add('done'));
}

// ── Error ──────────────────────────────────────────────────────────────────
function showError(msg) {
  errorMsg.textContent = msg;
  errorBanner.classList.add('visible');
}
function hideError() {
  errorBanner.classList.remove('visible');
}

// ── File chip ──────────────────────────────────────────────────────────────
function showFileChip(name) {
  fileChipName.textContent = name;
  fileChip.classList.add('visible');
}
function hideFileChip() {
  fileChip.classList.remove('visible');
  fileInput.value = '';
  currentFile = null;
}

fileChipRemove.addEventListener('click', hideFileChip);
attachBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  const f = fileInput.files[0];
  if (f) { currentFile = f; showFileChip(f.name); }
});

// ── Drag and drop on chat input ────────────────────────────────────────────
chatInputBox.addEventListener('dragover', e => { e.preventDefault(); chatInputBox.classList.add('drag-over'); });
chatInputBox.addEventListener('dragleave', () => chatInputBox.classList.remove('drag-over'));
chatInputBox.addEventListener('drop', e => {
  e.preventDefault(); chatInputBox.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) {
    const dt = new DataTransfer(); dt.items.add(f);
    fileInput.files = dt.files; currentFile = f; showFileChip(f.name);
  }
});

// ── Auto-grow textarea ─────────────────────────────────────────────────────
mainInput.addEventListener('input', () => {
  mainInput.style.height = 'auto';
  mainInput.style.height = Math.min(mainInput.scrollHeight, 180) + 'px';
});

// Enter to submit (Shift+Enter = newline)
mainInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSubmit();
  }
});

// ── Input type detection ───────────────────────────────────────────────────
function detectType(value, hasFile) {
  if (hasFile) return 'file';
  if (/^https?:\/\//i.test(value.trim())) return 'url';
  return 'text';
}

// ── Submit ─────────────────────────────────────────────────────────────────
submitBtn.addEventListener('click', handleSubmit);

async function handleSubmit() {
  hideError();
  const value   = mainInput.value.trim();
  const hasFile = !!currentFile;

  if (!value && !hasFile) {
    showError('Please paste a recipe, enter a URL, or attach a file.');
    return;
  }

  const type     = detectType(value, hasFile);
  const formData = new FormData();
  if (type === 'file')      formData.append('file', currentFile);
  else if (type === 'url')  formData.append('url', value);
  else                       formData.append('text', value);

  setState('loading');
  startSteps();

  try {
    const res  = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();
    stopSteps();

    if (!res.ok || data.error) {
      setState('initial');
      showError(data.error ?? `Server error ${res.status}`);
      return;
    }

    const { recipe, predictions, confidence, dish_info } = data;
    buildDishInfo(dish_info, recipe.recipe_name);
    buildIngredientSidebar(recipe.ingredients);
    buildRecipeTable(recipe);
    buildScoresTable(predictions, confidence);
    buildRadar(predictions, confidence);
    renderConfidence(recipe.ingredients);
    document.getElementById('recipeName').textContent = recipe.recipe_name;

    setState('results');
  } catch (err) {
    stopSteps();
    setState('initial');
    showError(`Network error: ${err.message}`);
  }
}

// ── Confidence computation ─────────────────────────────────────────────────
function computeConfidence(ingredients) {
  const map = { High: 0.88, Medium: 0.55, Low: 0.25 };
  let wSum = 0, wConf = 0;
  ingredients.forEach(ing => {
    const w = ing.weight || 0;
    wSum   += w;
    wConf  += w * (ing.source === 'database' ? 1.0 : (map[ing.confidence] || 0.25));
  });
  return wSum > 0 ? Math.round((wConf / wSum) * 100) : 0;
}

function renderConfidence(ingredients) {
  const pct = computeConfidence(ingredients);
  document.getElementById('confidencePct').textContent = `${pct}%`;
  // Animate bar after paint
  requestAnimationFrame(() => {
    setTimeout(() => {
      document.getElementById('confidenceBar').style.width = `${pct}%`;
    }, 50);
  });
}

// ── Dish info (summary panel) ──────────────────────────────────────────────
function buildDishInfo(dish_info, recipe_name) {
  document.getElementById('dishName').textContent        = recipe_name;
  document.getElementById('dishDescription').textContent = dish_info?.description ?? '';

  const wrap = document.getElementById('dishImageWrap');
  if (dish_info?.image_url) {
    const img = document.createElement('img');
    img.src = dish_info.image_url;
    img.alt = recipe_name;
    img.onerror = () => {
      wrap.innerHTML = `<div class="dish-placeholder"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/></svg><span>Image unavailable</span></div>`;
    };
    wrap.innerHTML = '';
    wrap.appendChild(img);
  }
}

// ── Compact ingredient sidebar ─────────────────────────────────────────────
function buildIngredientSidebar(ingredients) {
  const list = document.getElementById('ingredientList');
  list.innerHTML = ingredients.map(ing => {
    const pct = Math.min(ing.weight * 100, 100);
    return `<div class="ing-row">
      <span class="ing-name">${ing.name}</span>
      <div class="ing-bar-wrap"><div class="ing-bar" style="width:${pct}%"></div></div>
      <span class="ing-pct">${(ing.weight * 100).toFixed(1)}%</span>
    </div>`;
  }).join('');
}

// ── Ingredient detail table ────────────────────────────────────────────────
function _sourceBadge(source, confidence) {
  if (source === 'database') return { label: 'Database', cls: 'badge-database' };
  if (source === 'predicted') {
    if (confidence === 'High')   return { label: 'Predicted · High',   cls: 'badge-predicted-high' };
    if (confidence === 'Medium') return { label: 'Predicted · Medium', cls: 'badge-predicted-medium' };
    return                              { label: 'Predicted · Low',    cls: 'badge-predicted-low' };
  }
  return { label: 'Unknown', cls: 'badge-unknown' };
}

function buildRecipeTable(recipe) {
  const tbody = document.getElementById('ingredientTbody');
  tbody.innerHTML = '';
  recipe.ingredients.forEach(ing => {
    const w = ing.weight, s = ing.sensory_scores;
    const row = document.createElement('tr');

    const tdN = document.createElement('td');
    tdN.innerHTML = `<span class="ingredient-name">${ing.name}</span>`;

    const tdW = document.createElement('td');
    tdW.className = 'weight-bar-cell';
    tdW.innerHTML = `<div class="weight-bar-wrap"><div class="weight-bar-bg"><div class="weight-bar" style="width:${Math.min(w*100,100)}%"></div></div><span class="weight-pct">${(w*100).toFixed(1)}%</span></div>`;

    const src   = ing.source || 'unknown';
    const conf  = ing.confidence || null;
    const badge = _sourceBadge(src, conf);
    const tdSrc = document.createElement('td');
    const title = src === 'database' && ing.matched_name ? `Matched to: ${ing.matched_name}` : '';
    tdSrc.innerHTML = `<span class="source-badge ${badge.cls}" title="${title}">${badge.label}</span>`;

    const tdConf = document.createElement('td');
    if (src === 'predicted' && conf) {
      const cls = conf === 'High' ? 'conf-high' : conf === 'Medium' ? 'conf-medium' : 'conf-low';
      tdConf.innerHTML = `<span class="conf-value ${cls}">${conf}</span>`;
    } else {
      tdConf.innerHTML = `<span style="color:var(--text-dim)">—</span>`;
    }

    const tdEv = document.createElement('td');
    if (src === 'predicted' && ing.evidence) {
      const ev = ing.evidence;
      const short = ev.length > 55 ? ev.slice(0, 55) + '…' : ev;
      // Linkify any URLs in the evidence
      const linked = ev.replace(/(https?:\/\/[^\s,;)"']+)/g,
        '<a href="$1" target="_blank" rel="noopener" class="evidence-link" onclick="event.stopPropagation()">↗</a>');
      const btn = document.createElement('span');
      btn.className = 'evidence-text';
      btn.title = 'Click to expand';
      btn.innerHTML = short;
      btn.addEventListener('click', () => showEvidenceModal(ev, linked));
      tdEv.appendChild(btn);
      // Append link icons inline if URLs present
      const urls = ev.match(/(https?:\/\/[^\s,;)"']+)/g) || [];
      urls.forEach(url => {
        const a = document.createElement('a');
        a.href = url; a.target = '_blank'; a.rel = 'noopener';
        a.className = 'evidence-link'; a.textContent = '↗';
        a.addEventListener('click', e => e.stopPropagation());
        tdEv.appendChild(a);
      });
    } else {
      tdEv.innerHTML = `<span style="color:var(--text-dim)">—</span>`;
    }

    const tdS = document.createElement('td');
    tdS.innerHTML = SENSES.map(sense => {
      const v = s && s[sense] != null ? Number(s[sense]).toFixed(0) : '–';
      return `<span class="sense-chip" style="background:${SENSE_COLORS[sense]}18;color:${SENSE_COLORS[sense]};border:1px solid ${SENSE_COLORS[sense]}40"><span class="chip-label">${SENSE_LABELS[sense]}</span><span class="chip-val">${v}</span></span>`;
    }).join('');

    row.append(tdN, tdW, tdSrc, tdConf, tdEv, tdS);
    tbody.appendChild(row);
  });
}

// ── Scores table ───────────────────────────────────────────────────────────
function buildScoresTable(predictions, confidence) {
  const c = document.getElementById('scoresContainer');
  c.innerHTML = '';
  SENSES.forEach(sense => {
    const pred = predictions[sense] ?? 0;
    const lo   = confidence[sense]?.lower ?? 0;
    const hi   = confidence[sense]?.upper ?? 0;
    const col  = SENSE_COLORS[sense];
    const row  = document.createElement('div');
    row.className = 'score-row';
    row.innerHTML = `
      <div class="score-sense-name"><div class="sense-indicator" style="background:${col}"></div>${SENSE_LABELS[sense]}</div>
      <div class="score-gauge-wrap"><div class="score-gauge-bg">
        <div class="score-gauge-range" style="left:${lo}%;width:${hi-lo}%;background:${col}"></div>
        <div class="score-gauge-fill"  style="width:${pred}%;background:${col}"></div>
      </div></div>
      <div class="score-values"><div class="score-main" style="color:${col}">${pred.toFixed(1)}</div><div class="score-range">${lo.toFixed(1)} – ${hi.toFixed(1)}</div></div>`;
    c.appendChild(row);
  });
}

// ── Evidence modal ─────────────────────────────────────────────────────────
function showEvidenceModal(text, linkedHtml) {
  let modal = document.getElementById('evidenceModal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'evidenceModal';
    modal.className = 'ev-modal-backdrop';
    modal.innerHTML = `<div class="ev-modal"><button class="ev-modal-close" id="evClose">×</button><div class="ev-modal-body" id="evBody"></div></div>`;
    document.body.appendChild(modal);
    modal.addEventListener('click', e => { if (e.target === modal) modal.style.display = 'none'; });
    document.getElementById('evClose').addEventListener('click', () => { modal.style.display = 'none'; });
  }
  document.getElementById('evBody').innerHTML = linkedHtml;
  modal.style.display = 'flex';
}

// ── Radar chart ────────────────────────────────────────────────────────────
function buildRadar(predictions, confidence) {
  const ctx = document.getElementById('radarCanvas').getContext('2d');
  if (radarChart) radarChart.destroy();
  const data   = SENSES.map(s => predictions[s] ?? 0);
  const upper  = SENSES.map(s => confidence[s]?.upper ?? 0);
  const labels = SENSES.map(s => SENSE_LABELS[s]);

  radarChart = new Chart(ctx, {
    type: 'radar',
    data: {
      labels,
      datasets: [
        { label: 'Range', data: upper, borderColor: 'transparent',
          backgroundColor: 'rgba(124,58,237,0.06)', pointRadius: 0, fill: true },
        { label: 'Predicted', data,
          borderColor: '#7C3AED', borderWidth: 2,
          backgroundColor: 'rgba(124,58,237,0.09)',
          pointBackgroundColor: SENSES.map(s => SENSE_COLORS[s]),
          pointBorderColor: '#fff', pointRadius: 5, pointHoverRadius: 7, fill: true },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: true,
      animation: { duration: 900, easing: 'easeOutQuart' },
      scales: { r: {
        min: 0, max: 100,
        ticks: { stepSize: 25, color: '#9CA3AF', font: { size: 9 }, backdropColor: 'transparent' },
        grid: { color: 'rgba(0,0,0,0.06)' },
        angleLines: { color: 'rgba(0,0,0,0.07)' },
        pointLabels: { font: { size: 12, family: "'Sora'", weight: '600' },
          color: SENSES.map(s => SENSE_COLORS[s]) },
      }},
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#fff', borderColor: '#E5E7EB', borderWidth: 1,
          titleColor: '#111827', bodyColor: '#6B7280', padding: 10,
          callbacks: { label: ctx => {
            const s = SENSES[ctx.dataIndex];
            if (ctx.datasetIndex === 1) {
              const lo = confidence[s]?.lower ?? 0, hi = confidence[s]?.upper ?? 0;
              return ` ${ctx.parsed.r.toFixed(1)}  (${lo.toFixed(1)}–${hi.toFixed(1)})`;
            }
            return '';
          }},
        },
      },
    },
  });
}
