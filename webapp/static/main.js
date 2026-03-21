/* FlavorLab – main.js */

const SENSES = ['sweet', 'bitter', 'salty', 'umami', 'sour'];
const SENSE_COLORS = { sweet:'#4477AA', bitter:'#EE6677', salty:'#228833', umami:'#CCBB44', sour:'#AA3377' };
const SENSE_LABELS = { sweet:'Sweet', bitter:'Bitter', salty:'Salty', umami:'Umami', sour:'Sour' };

const tabs       = document.querySelectorAll('.tab-btn');
const panes      = document.querySelectorAll('.tab-pane');
const textarea   = document.getElementById('recipeText');
const urlInput   = document.getElementById('recipeUrl');
const fileInput  = document.getElementById('fileInput');
const dropZone   = document.getElementById('dropZone');
const fileBadge  = document.getElementById('fileBadge');
const fileBadgeName = document.getElementById('fileBadgeName');
const predictBtn = document.getElementById('predictBtn');
const loadingEl  = document.getElementById('loadingOverlay');
const errorEl    = document.getElementById('errorBanner');
const errorMsg   = document.getElementById('errorMsg');
const resultsEl  = document.getElementById('resultsSection');
let radarChart   = null;

// ── Tabs ─────────────────────────────────────────────────────────────────────
tabs.forEach(btn => btn.addEventListener('click', () => {
  tabs.forEach(t  => t.classList.remove('active'));
  panes.forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(`pane-${btn.dataset.tab}`).classList.add('active');
}));

// ── File drop ─────────────────────────────────────────────────────────────────
function showFileBadge(name) { fileBadgeName.textContent = name; fileBadge.style.display = 'inline-flex'; }
fileInput.addEventListener('change', () => { if (fileInput.files[0]) showFileBadge(fileInput.files[0].name); });
dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) { const dt = new DataTransfer(); dt.items.add(f); fileInput.files = dt.files; showFileBadge(f.name); }
});

// ── Loading steps ─────────────────────────────────────────────────────────────
const stepEls = document.querySelectorAll('.loading-steps li');
let stepTimer = null;
function animateSteps() {
  let i = 0; stepEls.forEach(el => el.classList.remove('active','done'));
  function next() {
    if (i > 0) stepEls[i-1].classList.replace('active','done');
    if (i < stepEls.length) { stepEls[i].classList.add('active'); i++; stepTimer = setTimeout(next, 2200); }
  }
  next();
}
function stopSteps() { clearTimeout(stepTimer); stepEls.forEach(el => el.classList.remove('active','done')); }

// ── Error ─────────────────────────────────────────────────────────────────────
function showError(msg) {
  errorMsg.textContent = msg; errorEl.classList.add('visible');
  loadingEl.classList.remove('visible'); stopSteps(); predictBtn.disabled = false;
}
function hideError() { errorEl.classList.remove('visible'); }

// ── Radar chart ───────────────────────────────────────────────────────────────
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
        { label:'Range', data: upper, borderColor:'transparent',
          backgroundColor:'rgba(45,106,79,0.07)', pointRadius:0, fill:true },
        { label:'Predicted', data,
          borderColor:'#2D6A4F', borderWidth:2,
          backgroundColor:'rgba(45,106,79,0.10)',
          pointBackgroundColor: SENSES.map(s => SENSE_COLORS[s]),
          pointBorderColor:'#fff', pointRadius:5, pointHoverRadius:7, fill:true },
      ],
    },
    options: {
      responsive:true, maintainAspectRatio:true,
      animation:{ duration:900, easing:'easeOutQuart' },
      scales: { r: {
        min:0, max:100,
        ticks:{ stepSize:25, color:'#96AEA5', font:{size:9}, backdropColor:'transparent' },
        grid:{ color:'rgba(0,0,0,0.06)' },
        angleLines:{ color:'rgba(0,0,0,0.07)' },
        pointLabels:{ font:{size:12, family:"'Fraunces'", weight:'600'},
          color: SENSES.map(s => SENSE_COLORS[s]) },
      }},
      plugins: {
        legend:{ display:false },
        tooltip:{
          backgroundColor:'#fff', borderColor:'#DDE6E1', borderWidth:1,
          titleColor:'#192B22', bodyColor:'#5C7268', padding:10,
          callbacks:{ label: ctx => {
            const s = SENSES[ctx.dataIndex];
            if (ctx.datasetIndex===1) {
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

// ── Recipe table ──────────────────────────────────────────────────────────────
const SOURCE_BADGE = {
  database:   { label: 'Database',   cls: 'badge-database' },
  researched: { label: 'Researched', cls: 'badge-researched' },
  unknown:    { label: 'Unknown',    cls: 'badge-unknown' },
};

function buildRecipeTable(recipe) {
  document.getElementById('recipeName').textContent = recipe.recipe_name;
  const tbody = document.getElementById('ingredientTbody');
  tbody.innerHTML = '';
  recipe.ingredients.forEach(ing => {
    const w = ing.weight, s = ing.sensory_scores, row = document.createElement('tr');

    // Ingredient name
    const tdN = document.createElement('td');
    tdN.innerHTML = `<span class="ingredient-name">${ing.name}</span>`;

    // Weight bar
    const tdW = document.createElement('td');
    tdW.className = 'weight-bar-cell';
    tdW.innerHTML = `<div class="weight-bar-wrap"><div class="weight-bar-bg"><div class="weight-bar" style="width:${Math.min(w*100,100)}%"></div></div><span class="weight-pct">${(w*100).toFixed(1)}%</span></div>`;

    // Source badge
    const src   = ing.source || 'unknown';
    const badge = SOURCE_BADGE[src] || SOURCE_BADGE.unknown;
    const tdSrc = document.createElement('td');
    let srcTitle = '';
    if (src === 'database' && ing.matched_name) srcTitle = `Matched to: ${ing.matched_name}`;
    tdSrc.innerHTML = `<span class="source-badge ${badge.cls}" title="${srcTitle}">${badge.label}</span>`;

    // Confidence
    const tdConf = document.createElement('td');
    if (src === 'researched' && ing.confidence != null) {
      const pct = Math.round(ing.confidence * 100);
      tdConf.innerHTML = `<span class="conf-value">${pct}%</span>`;
    } else {
      tdConf.innerHTML = `<span style="color:var(--text-dim)">—</span>`;
    }

    // Evidence
    const tdEv = document.createElement('td');
    if (src === 'researched' && ing.evidence) {
      const url    = ing.source_url ? ` <a href="${ing.source_url}" target="_blank" rel="noopener" class="evidence-link">↗</a>` : '';
      tdEv.innerHTML = `<span class="evidence-text" title="${ing.evidence}">${ing.evidence.slice(0,60)}${ing.evidence.length>60?'…':''}${url}</span>`;
    } else {
      tdEv.innerHTML = `<span style="color:var(--text-dim)">—</span>`;
    }

    // Sensory chips
    const tdS = document.createElement('td');
    tdS.innerHTML = SENSES.map(sense => {
      const v = s && s[sense] != null ? Number(s[sense]).toFixed(0) : '–';
      return `<span class="sense-chip" style="background:${SENSE_COLORS[sense]}18;color:${SENSE_COLORS[sense]};border:1px solid ${SENSE_COLORS[sense]}40"><span class="chip-label">${SENSE_LABELS[sense]}</span><span class="chip-val">${v}</span></span>`;
    }).join('');

    row.append(tdN, tdW, tdSrc, tdConf, tdEv, tdS);
    tbody.appendChild(row);
  });
}

// ── Scores table ──────────────────────────────────────────────────────────────
function buildScoresTable(predictions, confidence) {
  const c = document.getElementById('scoresContainer');
  c.innerHTML = '';
  SENSES.forEach(sense => {
    const pred = predictions[sense]??0, lo = confidence[sense]?.lower??0, hi = confidence[sense]?.upper??0, col = SENSE_COLORS[sense];
    const row = document.createElement('div');
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

// ── Dish info ─────────────────────────────────────────────────────────────────
function buildDishInfo(dish_info, recipe_name) {
  const card = document.getElementById('dishInfoCard');
  if (!dish_info) { card.style.display = 'none'; return; }
  card.style.display = 'block';
  document.getElementById('dishName').textContent = recipe_name;
  document.getElementById('dishDescription').textContent = dish_info.description ?? '';
  const wrap = document.getElementById('dishImageWrap');
  if (dish_info.image_url) {
    const img = document.createElement('img');
    img.src = dish_info.image_url;
    img.alt = recipe_name;
    img.onerror = () => { wrap.innerHTML = `<div class="dish-placeholder"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/></svg><span>Image unavailable</span></div>`; };
    wrap.innerHTML = ''; wrap.appendChild(img);
  }
}

// ── Submit ────────────────────────────────────────────────────────────────────
predictBtn.addEventListener('click', async () => {
  hideError();
  const activeTab = document.querySelector('.tab-btn.active').dataset.tab;
  const formData  = new FormData();

  if (activeTab === 'text') {
    const txt = textarea.value.trim();
    if (!txt) { showError('Please paste a recipe description first.'); return; }
    formData.append('text', txt);
  } else if (activeTab === 'url') {
    const url = urlInput.value.trim();
    if (!url) { showError('Please enter a recipe URL.'); return; }
    formData.append('url', url);
  } else {
    if (!fileInput.files[0]) { showError('Please select a file to upload.'); return; }
    formData.append('file', fileInput.files[0]);
  }

  predictBtn.disabled = true;
  resultsEl.classList.remove('visible');
  loadingEl.classList.add('visible');
  animateSteps();

  try {
    const res  = await fetch('/predict', { method:'POST', body:formData });
    const data = await res.json();
    stopSteps(); loadingEl.classList.remove('visible'); predictBtn.disabled = false;

    if (!res.ok || data.error) { showError(data.error ?? `Server error: ${res.status}`); return; }

    const { recipe, predictions, confidence, dish_info } = data;
    buildDishInfo(dish_info, recipe.recipe_name);
    buildRecipeTable(recipe);
    buildScoresTable(predictions, confidence);
    buildRadar(predictions, confidence);
    document.getElementById('estimateBanner').style.display = 'none';
    resultsEl.classList.add('visible');
    resultsEl.scrollIntoView({ behavior:'smooth', block:'start' });
  } catch (err) {
    showError(`Network error: ${err.message}`);
  }
});

textarea.addEventListener('keydown', e => { if (e.key==='Enter' && e.metaKey) predictBtn.click(); });
