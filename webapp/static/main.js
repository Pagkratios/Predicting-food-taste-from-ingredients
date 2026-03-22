/* FlavorLab – main.js */

const SENSES      = ['sweet', 'bitter', 'salty', 'umami', 'sour'];
const SENSE_COLORS = { sweet:'#4477AA', bitter:'#EE6677', salty:'#228833', umami:'#CCBB44', sour:'#AA3377' };
const SENSE_LABELS = { sweet:'Sweet', bitter:'Bitter', salty:'Salty', umami:'Umami', sour:'Sour' };

// ── DOM refs ──────────────────────────────────────────────────────────────────
const mainInput      = document.getElementById('mainInput');
const submitBtn      = document.getElementById('submitBtn');
const attachBtn      = document.getElementById('attachBtn');
const fileInput      = document.getElementById('fileInput');
const fileChip       = document.getElementById('fileChip');
const fileChipName   = document.getElementById('fileChipName');
const fileChipRemove = document.getElementById('fileChipRemove');
const recipeInputBox = document.getElementById('recipeInputBox');
const centerView     = document.getElementById('centerView');
const heroContent    = document.getElementById('heroContent');
const stepsPanel     = document.getElementById('stepsPanel');
const stepListEl     = document.getElementById('stepList');
const resultsLayout  = document.getElementById('resultsLayout');
const errorBanner    = document.getElementById('errorBanner');
const errorMsg       = document.getElementById('errorMsg');
const chatMessages   = document.getElementById('chatMessages');
const botInput       = document.getElementById('botInput');
const chatSendBtn    = document.getElementById('chatSendBtn');
const newChatBtn     = document.getElementById('newChatBtn');

let radarChart     = null;
let stepTimer      = null;
let currentFile    = null;

// ── Session state ─────────────────────────────────────────────────────────────
let sessionId          = null;
let currentRecipe      = null;
let currentPredictions = null;
let currentConfidence  = null;
let currentDishInfo    = null;
let chatHistory        = [];  // [{role:'user'|'assistant', text:'...'}]

// ── Welcome HTML (reused when starting a new chat) ───────────────────────────
const WELCOME_HTML = `
  <div class="chat-welcome-icon">
    <svg viewBox="0 0 24 24"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
  </div>
  <p class="chat-welcome-title">FlavorLab Assistant</p>
  <p class="chat-welcome-sub">Submit a recipe on the right to get started. I can help you explore how changing ingredient proportions affects the predicted sensory profile.</p>
  <ul class="chat-welcome-hints">
    <li>"Increase sugar to 25%"</li>
    <li>"Remove the butter"</li>
    <li>"Add 10% olive oil"</li>
    <li>"Why is this so bitter?"</li>
  </ul>`;

// ── UUID helper ───────────────────────────────────────────────────────────────
function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
  });
}

// ── Session persistence ───────────────────────────────────────────────────────
function initSession() {
  sessionId = localStorage.getItem('flavorlab_session_id');
  if (!sessionId) {
    sessionId = generateUUID();
    localStorage.setItem('flavorlab_session_id', sessionId);
  }

  const savedState = localStorage.getItem('flavorlab_state');
  if (savedState === 'results') {
    try {
      const r = localStorage.getItem('flavorlab_recipe');
      const p = localStorage.getItem('flavorlab_predictions');
      const c = localStorage.getItem('flavorlab_confidence');
      const d = localStorage.getItem('flavorlab_dish_info');
      const h = localStorage.getItem('flavorlab_chat');
      if (r && p && c) {
        currentRecipe      = JSON.parse(r);
        currentPredictions = JSON.parse(p);
        currentConfidence  = JSON.parse(c);
        currentDishInfo    = d ? JSON.parse(d) : null;
        chatHistory        = h ? JSON.parse(h) : [];
        restoreResults();
      }
    } catch (_) {
      // Corrupted storage — start fresh
      clearLocalStorage();
    }
  }
}

function saveSession() {
  localStorage.setItem('flavorlab_state',       'results');
  localStorage.setItem('flavorlab_recipe',      JSON.stringify(currentRecipe));
  localStorage.setItem('flavorlab_predictions', JSON.stringify(currentPredictions));
  localStorage.setItem('flavorlab_confidence',  JSON.stringify(currentConfidence));
  if (currentDishInfo) {
    localStorage.setItem('flavorlab_dish_info', JSON.stringify(currentDishInfo));
  }
  localStorage.setItem('flavorlab_chat', JSON.stringify(chatHistory));

  // Persist to server (fire-and-forget backup)
  fetch('/session/save', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({
      session_id: sessionId,
      state: {
        recipe:       currentRecipe,
        predictions:  currentPredictions,
        confidence:   currentConfidence,
        dish_info:    currentDishInfo,
        chat_history: chatHistory,
      },
    }),
  }).catch(() => {});
}

function clearLocalStorage() {
  const keys = [
    'flavorlab_state', 'flavorlab_recipe', 'flavorlab_predictions',
    'flavorlab_confidence', 'flavorlab_dish_info', 'flavorlab_chat',
  ];
  keys.forEach(k => localStorage.removeItem(k));
}

function restoreResults() {
  buildDishInfo(currentDishInfo, currentRecipe.recipe_name);
  buildIngredientFractions(currentRecipe.ingredients);
  buildRecipeTable(currentRecipe);
  buildScoresTable(currentPredictions, currentConfidence);
  buildRadar(currentPredictions, currentConfidence);
  renderConfidence(currentRecipe.ingredients);
  document.getElementById('recipeName').textContent = currentRecipe.recipe_name;

  // Restore chat messages
  chatHistory.forEach(msg => _appendBubble(msg.role, msg.text));

  setState('results');
}

// ── New chat ──────────────────────────────────────────────────────────────────
function newChat() {
  clearLocalStorage();
  sessionId = generateUUID();
  localStorage.setItem('flavorlab_session_id', sessionId);

  currentRecipe      = null;
  currentPredictions = null;
  currentConfidence  = null;
  currentDishInfo    = null;
  chatHistory        = [];

  // Reset chat panel
  chatMessages.innerHTML = '';
  const welcome = document.createElement('div');
  welcome.id        = 'chatWelcome';
  welcome.className = 'chat-welcome';
  welcome.innerHTML = WELCOME_HTML;
  chatMessages.appendChild(welcome);

  // Reset recipe input
  mainInput.value = '';
  hideFileChip();
  hideError();

  setState('initial');
}

newChatBtn.addEventListener('click', newChat);

// ── State machine ─────────────────────────────────────────────────────────────
function setState(state) {
  document.body.className = `state-${state}`;

  const chatEnabled = state === 'results';
  botInput.disabled   = !chatEnabled;
  chatSendBtn.disabled = !chatEnabled;
  botInput.placeholder = chatEnabled ? 'Ask about the recipe...' : 'Submit a recipe to get started...';

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
    heroContent.style.transition = 'opacity 0.25s ease, transform 0.3s ease';
    heroContent.style.opacity    = '0';
    heroContent.style.transform  = 'translateY(-14px)';
    setTimeout(() => {
      if (document.body.classList.contains('state-loading')) {
        heroContent.style.display = 'none';
      }
    }, 310);

    centerView.style.display   = '';
    centerView.style.opacity   = '1';
    stepsPanel.style.display   = 'block';
    stepsPanel.style.animation = 'none';
    requestAnimationFrame(() => { stepsPanel.style.animation = ''; });
    resultsLayout.style.display = 'none';
    submitBtn.disabled = true;
  }

  if (state === 'results') {
    centerView.style.transition = 'opacity 0.25s ease';
    centerView.style.opacity    = '0';
    setTimeout(() => {
      if (document.body.classList.contains('state-results')) {
        centerView.style.display = 'none';
      }
    }, 270);

    resultsLayout.style.display    = 'block';
    resultsLayout.style.opacity    = '0';
    resultsLayout.style.transition = '';
    requestAnimationFrame(() => {
      resultsLayout.style.transition = 'opacity 0.4s ease';
      resultsLayout.style.opacity    = '1';
    });
    submitBtn.disabled = false;
  }
}

// ── Step animation ────────────────────────────────────────────────────────────
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

// ── Error ─────────────────────────────────────────────────────────────────────
function showError(msg) {
  errorMsg.textContent = msg;
  errorBanner.classList.add('visible');
}
function hideError() {
  errorBanner.classList.remove('visible');
}

// ── File chip ─────────────────────────────────────────────────────────────────
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

// ── Drag and drop on recipe input ─────────────────────────────────────────────
recipeInputBox.addEventListener('dragover', e => { e.preventDefault(); recipeInputBox.classList.add('drag-over'); });
recipeInputBox.addEventListener('dragleave', () => recipeInputBox.classList.remove('drag-over'));
recipeInputBox.addEventListener('drop', e => {
  e.preventDefault(); recipeInputBox.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) {
    const dt = new DataTransfer(); dt.items.add(f);
    fileInput.files = dt.files; currentFile = f; showFileChip(f.name);
  }
});

// ── Auto-grow textareas ───────────────────────────────────────────────────────
function autoGrow(el, max) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, max) + 'px';
}
mainInput.addEventListener('input', () => autoGrow(mainInput, 180));
botInput.addEventListener('input',  () => autoGrow(botInput,  120));

// Recipe input: Enter to submit, Shift+Enter = newline
mainInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
});

// Bot input: Enter to send, Shift+Enter = newline
botInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChatMessage(); }
});

// ── Input type detection ──────────────────────────────────────────────────────
function detectType(value, hasFile) {
  if (hasFile) return 'file';
  if (/^https?:\/\//i.test(value.trim())) return 'url';
  return 'text';
}

// ── Recipe submit ─────────────────────────────────────────────────────────────
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
  if (type === 'file')     formData.append('file', currentFile);
  else if (type === 'url') formData.append('url', value);
  else                     formData.append('text', value);

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

    // Store current state
    currentRecipe      = recipe;
    currentPredictions = predictions;
    currentConfidence  = confidence;
    currentDishInfo    = dish_info;
    chatHistory        = [];

    // Render results
    buildDishInfo(dish_info, recipe.recipe_name);
    buildIngredientFractions(recipe.ingredients);
    buildRecipeTable(recipe);
    buildScoresTable(predictions, confidence);
    buildRadar(predictions, confidence);
    renderConfidence(recipe.ingredients);
    document.getElementById('recipeName').textContent = recipe.recipe_name;

    // Add assistant welcome message in chat
    const summary = `I've analysed **${recipe.recipe_name}** — ${recipe.ingredients.length} ingredients found. You can ask me to change proportions, add or remove ingredients, or explain the predictions.`;
    _appendBubble('assistant', summary);
    chatHistory.push({ role: 'assistant', text: summary });

    saveSession();
    setState('results');

    // Clear recipe input
    mainInput.value = '';
    mainInput.style.height = '';
    hideFileChip();

  } catch (err) {
    stopSteps();
    setState('initial');
    showError(`Network error: ${err.message}`);
  }
}

// ── Chat message handling ─────────────────────────────────────────────────────
chatSendBtn.addEventListener('click', sendChatMessage);

async function sendChatMessage() {
  const text = botInput.value.trim();
  if (!text || botInput.disabled) return;

  botInput.value = '';
  botInput.style.height = '';

  _appendBubble('user', text);
  chatHistory.push({ role: 'user', text });

  // Typing indicator
  const typing = document.createElement('div');
  typing.className = 'chat-bubble assistant typing';
  typing.innerHTML = '<span class="typing-dots"><span></span><span></span><span></span></span>';
  chatMessages.appendChild(typing);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  chatSendBtn.disabled = true;
  botInput.disabled    = true;

  try {
    const res  = await fetch('/chat', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        session_id:   sessionId,
        message:      text,
        recipe:       currentRecipe,
        chat_history: chatHistory.slice(0, -1), // exclude the message we just pushed
      }),
    });
    const data = await res.json();
    typing.remove();

    if (data.error) {
      _appendBubble('assistant', `Sorry, something went wrong: ${data.error}`);
      return;
    }

    _appendBubble('assistant', data.reply);
    chatHistory.push({ role: 'assistant', text: data.reply });

    if (data.action === 'modify_recipe' && data.recipe) {
      currentRecipe      = data.recipe;
      currentPredictions = data.predictions;
      currentConfidence  = data.confidence;

      buildDishInfo(currentDishInfo, currentRecipe.recipe_name);
      buildIngredientFractions(currentRecipe.ingredients);
      buildRecipeTable(currentRecipe);
      buildScoresTable(currentPredictions, currentConfidence);
      buildRadar(currentPredictions, currentConfidence);
      renderConfidence(currentRecipe.ingredients);
    }

    saveSession();
  } catch (err) {
    typing.remove();
    _appendBubble('assistant', `Network error: ${err.message}`);
  } finally {
    chatSendBtn.disabled = false;
    botInput.disabled    = false;
    botInput.focus();
  }
}

// ── Chat bubble renderer ──────────────────────────────────────────────────────
function _appendBubble(role, text) {
  // Remove welcome card on first real message
  const welcome = document.getElementById('chatWelcome');
  if (welcome) welcome.remove();

  const div = document.createElement('div');
  div.className = `chat-bubble ${role}`;
  // Render **bold** markdown
  div.innerHTML = text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ── Confidence computation ────────────────────────────────────────────────────
function computeConfidence(ingredients) {
  const map = { High: 0.88, Medium: 0.55, Low: 0.25 };
  let wSum = 0, wConf = 0;
  ingredients.forEach(ing => {
    const w = ing.weight || 0;
    wSum  += w;
    wConf += w * (ing.source === 'database' ? 1.0 : (map[ing.confidence] || 0.25));
  });
  return wSum > 0 ? Math.round((wConf / wSum) * 100) : 0;
}

function renderConfidence(ingredients) {
  const pct = computeConfidence(ingredients);
  document.getElementById('confidencePct').textContent = `${pct}%`;
  requestAnimationFrame(() => {
    setTimeout(() => {
      document.getElementById('confidenceBar').style.width = `${pct}%`;
    }, 50);
  });
}

// ── Dish info (top card) ──────────────────────────────────────────────────────
function buildDishInfo(dish_info, recipe_name) {
  document.getElementById('dishName').textContent        = recipe_name;
  document.getElementById('dishDescription').textContent = dish_info?.description ?? '';

  const wrap = document.getElementById('dishImageWrap');
  if (dish_info?.image_url) {
    const img = document.createElement('img');
    img.src       = dish_info.image_url;
    img.alt       = recipe_name;
    img.className = 'dish-image-thumb';
    img.onerror   = () => {
      wrap.innerHTML = `<div class="dish-placeholder"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/></svg><span>Image unavailable</span></div>`;
    };
    wrap.innerHTML = '';
    wrap.appendChild(img);
  } else {
    wrap.innerHTML = `<div class="dish-placeholder"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/></svg><span>No image found</span></div>`;
  }
}

// ── Ingredient fraction list (top card) ───────────────────────────────────────
function buildIngredientFractions(ingredients) {
  const list = document.getElementById('ingredientFractionList');
  list.innerHTML = ingredients.map(ing => {
    const pct = (ing.weight * 100).toFixed(1);
    const dot = ing.source === 'database' ? '#059669' : '#6B7280';
    return `<div class="fraction-item">
      <span class="fraction-dot" style="background:${dot}"></span>
      <span class="fraction-name">${ing.name}</span>
      <span class="fraction-pct">${pct}%</span>
    </div>`;
  }).join('');
}

// ── Ingredient detail table ───────────────────────────────────────────────────
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
      const ev    = ing.evidence;
      const short = ev.length > 55 ? ev.slice(0, 55) + '…' : ev;
      const linked = ev.replace(/(https?:\/\/[^\s,;)"']+)/g,
        '<a href="$1" target="_blank" rel="noopener" class="evidence-link" onclick="event.stopPropagation()">↗</a>');
      const btn = document.createElement('span');
      btn.className = 'evidence-text';
      btn.title     = 'Click to expand';
      btn.innerHTML = short;
      btn.addEventListener('click', () => showEvidenceModal(ev, linked));
      tdEv.appendChild(btn);
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

// ── Scores table ──────────────────────────────────────────────────────────────
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

// ── Evidence modal ────────────────────────────────────────────────────────────
function _parseCitations(text) {
  const seen = new Set(), out = [];
  const doiRe = /\bDOI:?\s*(10\.\d{4,}\/\S+)/gi;
  let m;
  while ((m = doiRe.exec(text)) !== null) {
    const doi  = m[1].replace(/[.,;)"']+$/, '');
    const href = `https://doi.org/${doi}`;
    if (!seen.has(doi)) { seen.add(doi); out.push({ href, label: doi }); }
  }
  const urlRe = /https?:\/\/[^\s,;)"'\]]+/g;
  while ((m = urlRe.exec(text)) !== null) {
    const href = m[0].replace(/[.,;)"']+$/, '');
    if (!seen.has(href)) { seen.add(href); out.push({ href, label: href }); }
  }
  return out;
}

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
  const citations = _parseCitations(text);
  let html = `<div class="ev-main-text">${linkedHtml}</div>`;
  if (citations.length > 0) {
    html += `<div class="ev-citations"><div class="ev-citations-label">Sources</div>
      ${citations.map((c, i) => `<div class="ev-citation"><span class="ev-cit-num">${i+1}</span><a href="${c.href}" target="_blank" rel="noopener" class="ev-cit-link">${c.label}</a></div>`).join('')}
    </div>`;
  }
  document.getElementById('evBody').innerHTML = html;
  modal.style.display = 'flex';
}

// ── Radar scroll zoom ─────────────────────────────────────────────────────────
let radarMaxScale = 100;

document.getElementById('radarCanvas').addEventListener('wheel', e => {
  e.preventDefault();
  if (!radarChart) return;
  radarMaxScale = Math.min(100, Math.max(20, radarMaxScale + (e.deltaY > 0 ? 10 : -10)));
  radarChart.options.scales.r.max = radarMaxScale;
  radarChart.update('none');
}, { passive: false });

// ── Radar chart ───────────────────────────────────────────────────────────────
function buildRadar(predictions, confidence) {
  radarMaxScale = 100;
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

// ── Init ──────────────────────────────────────────────────────────────────────
initSession();
