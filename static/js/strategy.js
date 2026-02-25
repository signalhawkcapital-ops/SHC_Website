/* ── strategy.js — Batman Strategy Builder ── */

// Defensive: ensure showToast exists (may be defined in base template)
if (typeof showToast === 'undefined') {
  window.showToast = function(msg, type) {
    const el = document.createElement('div');
    el.style.cssText = 'position:fixed;top:20px;right:20px;z-index:9999;padding:12px 20px;border-radius:8px;font-size:13px;font-weight:600;max-width:400px;animation:fadeIn .2s;' +
      (type === 'error' ? 'background:#dc2626;color:#fff;' : 'background:var(--accent,#4f6ef7);color:#fff;');
    el.textContent = msg;
    document.body.appendChild(el);
    setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity .3s'; setTimeout(() => el.remove(), 300); }, 4000);
  };
}

const SESSION_KEY = 'batman_strategy_state';

let currentRisk = 'moderate';
let strategyData = null;

function setRisk(r) {
  currentRisk = r;
  document.querySelectorAll('.toggle-group button').forEach(b => b.classList.remove('active'));
  document.getElementById('risk-' + r).classList.add('active');
  // Auto-rerun if strategy already generated
  if (strategyData) runAnalysis();
}

/* ── Session persistence ── */
function saveState() {
  if (!strategyData) return;
  const state = {
    // inputs
    inputs: {
      spxOpen: document.getElementById('spxOpen').value,
      vixLevel: document.getElementById('vixLevel').value,
      expDate: document.getElementById('expDate').value,
      entryTime: document.getElementById('entryTime').value,
      expectedMove: document.getElementById('expectedMove').value,
      marketRegime: document.getElementById('marketRegime').value,
      risk: currentRisk,
    },
    // API response
    data: strategyData,
    // User-entered debits
    debits: {},
    // Timestamp
    savedAt: Date.now(),
  };
  // Capture all debit inputs
  document.querySelectorAll('.debit-input').forEach(el => {
    if (el.id && parseFloat(el.value)) state.debits[el.id] = el.value;
  });
  try { sessionStorage.setItem(SESSION_KEY, JSON.stringify(state)); } catch(e) {}
}

function loadState() {
  try {
    const raw = sessionStorage.getItem(SESSION_KEY);
    if (!raw) return false;
    const state = JSON.parse(raw);
    // Only restore if saved today (0DTE = same-day positions)
    const savedDate = new Date(state.savedAt).toDateString();
    const today = new Date().toDateString();
    if (savedDate !== today) { sessionStorage.removeItem(SESSION_KEY); return false; }
    if (!state.data) return false;

    // Restore inputs
    const inp = state.inputs || {};
    if (inp.spxOpen) document.getElementById('spxOpen').value = inp.spxOpen;
    if (inp.vixLevel) document.getElementById('vixLevel').value = inp.vixLevel;
    if (inp.expDate) document.getElementById('expDate').value = inp.expDate;
    if (inp.entryTime) document.getElementById('entryTime').value = inp.entryTime;
    if (inp.expectedMove) document.getElementById('expectedMove').value = inp.expectedMove;
    if (inp.marketRegime) document.getElementById('marketRegime').value = inp.marketRegime;
    if (inp.risk) setRisk(inp.risk);

    // Restore strategy data and render
    strategyData = state.data;
    window.strategyData = strategyData;
    if (!strategyData._inner_gap && strategyData._inner_gap !== 0) {
      strategyData._inner_gap = Math.max(0, Math.round((strategyData.center_gap || 0) / 10) * 10);
    }
    renderStrategy(strategyData);

    // Restore debits after render (needs slight delay for DOM)
    setTimeout(() => {
      const debits = state.debits || {};
      Object.entries(debits).forEach(([id, val]) => {
        const el = document.getElementById(id);
        if (el) el.value = val;
      });
      redrawPnl();
    }, 200);

    return true;
  } catch(e) {
    return false;
  }
}

function resetStrategy() {
  sessionStorage.removeItem(SESSION_KEY);
  strategyData = null;
  if (window._thetaRefreshInterval) clearInterval(window._thetaRefreshInterval);
  document.getElementById('results').classList.add('hidden');
  // Clear debits
  document.querySelectorAll('.debit-input').forEach(el => el.value = '0.00');
  // Reset inputs to defaults
  document.getElementById('expectedMove').value = '';
  document.getElementById('marketRegime').value = 'auto';
  document.getElementById('dowAdj').value = 'auto';
  setRisk('moderate');
  // Dismiss any trap alert
  const ab = document.getElementById('trapAlertBanner');
  if (ab) ab.style.display = 'none';
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

async function runAnalysis() {
  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Analyzing…';

  const payload = {
    spx_open: parseFloat(document.getElementById('spxOpen').value),
    vix: parseFloat(document.getElementById('vixLevel').value),
    trade_date: document.getElementById('expDate').value,
    entry_time: document.getElementById('entryTime').value,
    expected_move: document.getElementById('expectedMove').value || null,
    risk_profile: currentRisk,
    regime: document.getElementById('marketRegime').value,
  };

  // Client-side validation
  if (!payload.spx_open || payload.spx_open < 100 || payload.spx_open > 50000) {
    showToast('Enter a valid SPX open price', 'error');
    btn.disabled = false; btn.innerHTML = '⚡ Analyze &amp; Generate Strategy'; return;
  }
  if (!payload.vix || payload.vix < 0.5 || payload.vix > 150) {
    showToast('Enter a valid VIX level', 'error');
    btn.disabled = false; btn.innerHTML = '⚡ Analyze &amp; Generate Strategy'; return;
  }
  if (!payload.trade_date) {
    showToast('Select an expiration date', 'error');
    btn.disabled = false; btn.innerHTML = '⚡ Analyze &amp; Generate Strategy'; return;
  }

  try {
    const resp = await fetch('/api/strategy', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (resp.status === 401) { window.location.href = '/login?next=/batman'; return; }
    if (resp.status === 403) { window.location.href = '/account'; return; }
    const json = await resp.json();
    if (json.status !== 'ok') throw new Error(json.message);
    strategyData = json.data;
    window.strategyData = strategyData;  // expose for live trap alert system
    // Compute inner gap from the strikes: call_lower - put_upper
    strategyData.center_gap = strategyData.call_lower - strategyData.put_upper;
    strategyData._inner_gap = Math.max(0, Math.round(strategyData.center_gap / 10) * 10);
    strategyData._recommended_width = strategyData.butterfly_width;  // save original for confidence calc
    // Recompute pin/confidence with our client-side model
    _updateDashboardMetrics(strategyData);
    renderStrategy(strategyData);
    saveState();
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '⚡ Analyze &amp; Generate Strategy';
  }
}

/* ── Copy with feedback ── */
function copyText(chainId, btnEl) {
  const text = document.getElementById(chainId).textContent.trim();
  navigator.clipboard.writeText(text).then(() => {
    const origText = btnEl.textContent;
    btnEl.textContent = '✓ Copied';
    btnEl.style.background = 'var(--green)';
    btnEl.style.color = '#fff';
    btnEl.style.borderColor = 'var(--green)';
    const chainEl = document.getElementById(chainId);
    chainEl.style.borderColor = 'var(--green)';
    chainEl.style.boxShadow = '0 0 12px var(--green-dim)';
    setTimeout(() => {
      btnEl.textContent = origText;
      btnEl.style.background = '';
      btnEl.style.color = '';
      btnEl.style.borderColor = '';
      chainEl.style.borderColor = '';
      chainEl.style.boxShadow = '';
    }, 1500);
  });
}

/* ── Chain card builder ── */
function chainCard(title, titleColor, chainId, chainText, debitId) {
  return `
    <div class="card">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
        <h3 style="color:${titleColor};margin:0">${title}</h3>
        <button class="btn btn-secondary btn-sm copy-btn" onclick="copyText('${chainId}', this)">Copy</button>
      </div>
      <div class="chain-output" id="${chainId}">${chainText}</div>
      <div class="mt-3">
        <label>Debit Paid</label>
        <input type="number" id="${debitId}" class="debit-input" value="0.00" step="0.05" onchange="redrawPnl()">
      </div>
    </div>`;
}

/* ── Student-t CDF approximation (df=5) for pin probability ── */
// Uses the regularized incomplete beta function via rational approximation
function _tCDF(x, df) {
  // Convert t-value to probability using the relationship with Beta function
  // For the values we need (df=5, x from 0 to ~6), a numerical approach works well
  if (x === 0) return 0.5;
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t2 = x * x;
  // Use continued fraction / series for I_x(a,b) where a=df/2, b=1/2
  // Simpler: use the Hill (1970) approximation for small df
  const a = df / 2;
  const b = 0.5;
  const z = df / (df + t2);
  // Regularized incomplete beta I_z(a, b) via series
  let sum = 0, term = 1;
  for (let k = 0; k < 200; k++) {
    if (k === 0) { term = 1; }
    else { term *= (b + k - 1) * z / k; }
    // multiply by rising factorial of a
    let ak = 1;
    for (let j = 0; j < k; j++) ak *= (a + j) / (a + b + j);
    // Actually, let's use a simpler known formula for t-dist CDF
    break;
  }
  // Simpler: use normal approximation corrected for df
  // Cornish-Fisher expansion for t -> z
  const g1 = (t2 + 1) / (4 * df);
  const g2 = (5*t2*t2 + 16*t2 + 3) / (96 * df * df);
  const zn = x * (1 - g1 + g2);
  // Standard normal CDF
  const p = 0.5 * (1 + _erf(zn / Math.SQRT2));
  return sign > 0 ? p : 1 - p;
}

function _erf(x) {
  // Abramowitz & Stegun approximation 7.1.26 (max error 1.5e-7)
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * Math.exp(-x*x);
  return sign * y;
}

function _normCDF(x) {
  return 0.5 * (1 + _erf(x / Math.SQRT2));
}

function computePinProbability(width, gap, em) {
  /**
   * Probability settlement falls in profit zone.
   * Uses t-distribution (df=5) for fat tails.
   * scale = em / sqrt(5/3) since t(5) has variance = 5/3
   */
  if (em <= 0 || width <= 0) return 0.5;
  const df = 5;
  const scale = em / Math.sqrt(df / (df - 2));  // em / 1.291

  if (gap <= 0) {
    // Profit zone: |move| < 2×width
    const p = 2 * _tCDF(2 * width / scale, df) - 1;
    return Math.min(0.99, Math.max(0.1, p));
  } else {
    // Dead zone from -gap/2 to +gap/2
    const outer = gap/2 + 2 * width;
    const inner = gap/2;
    const oneSide = _tCDF(outer / scale, df) - _tCDF(inner / scale, df);
    return Math.min(0.99, Math.max(0.1, 2 * oneSide));
  }
}

function computeConfidence(width, gap, em, vix, recommendedWidth) {
  /**
   * Composite confidence score (0-99).
   * Updates dynamically as user adjusts width/gap.
   * 
   * Factors:
   *   Width deviation from recommendation (35%)
   *   Pin probability (25%)
   *   Regime/VIX (20%)
   *   Gap penalty (20%)
   */
  if (em <= 0) return 50;
  const recW = recommendedWidth || width;

  // 1. Width deviation from recommendation
  const ratio = width / recW;
  let widthScore;
  if (ratio >= 0.8 && ratio <= 1.2)
    widthScore = 95 - Math.abs(ratio - 1.0) * 50;
  else if (ratio >= 0.5 && ratio < 0.8)
    widthScore = 70 - (0.8 - ratio) / 0.3 * 30;
  else if (ratio > 1.2 && ratio <= 1.8)
    widthScore = 70 - (ratio - 1.2) / 0.6 * 25;
  else
    widthScore = 35;

  // 2. Pin probability
  const pin = computePinProbability(width, gap, em);
  const pinScore = pin * 100;

  // 3. Regime
  let regimeScore;
  if (vix < 14) regimeScore = 90;
  else if (vix < 18) regimeScore = 78;
  else if (vix < 22) regimeScore = 65;
  else if (vix < 28) regimeScore = 52;
  else regimeScore = 38;

  // 4. Gap penalty
  let gapScore = 100;
  if (gap > 0 && em > 0) {
    gapScore = Math.max(15, 100 - (gap / em) * 250);
  }

  const conf = widthScore * 0.35 + pinScore * 0.25 +
               regimeScore * 0.20 + gapScore * 0.20;
  return Math.round(Math.min(99, Math.max(10, conf)));
}

function _updateDashboardMetrics(d) {
  /**
   * Recalculate and update confidence + pin probability in the DOM.
   * Called from _recalcStrikes whenever width or gap changes.
   */
  const em = d.expected_move || 0;
  const vix = d.vix || 17;
  const gap = Math.max(0, d._inner_gap || d.center_gap || 0);
  const recW = d._recommended_width || d.butterfly_width;

  const pin = computePinProbability(d.butterfly_width, gap, em);
  const conf = computeConfidence(d.butterfly_width, gap, em, vix, recW);

  d.pin_probability = Math.round(pin * 100);
  d.confidence_score = conf;

  // Update DOM if elements exist
  const confEl = document.getElementById('confValue');
  const pinEl = document.getElementById('pinValue');
  const maxEl = document.getElementById('maxProfitValue');
  const confBar = document.getElementById('confBar');

  if (confEl) {
    const confColor = conf >= 70 ? 'green' : conf >= 50 ? 'amber' : 'red';
    confEl.textContent = conf + '%';
    confEl.className = 'val';
    confEl.parentElement.className = 'metric ' + confColor;
    if (confBar) {
      const grade = conf >= 70 ? 'high' : conf >= 50 ? 'med' : 'low';
      confBar.style.width = conf + '%';
      confBar.className = 'fill confidence-' + grade;
    }
  }
  if (pinEl) pinEl.textContent = d.pin_probability + '%';
  if (maxEl) maxEl.textContent = '$' + d.max_profit_per_contract.toLocaleString();
}

/* ── Recalculate all strikes from centers + width, then re-render ── */
function _recalcStrikes() {
  const d = strategyData;
  if (!d) return;
  d.call_lower = d.call_center - d.butterfly_width;
  d.call_upper = d.call_center + d.butterfly_width;
  d.put_upper = d.put_center + d.butterfly_width;
  d.put_lower = d.put_center - d.butterfly_width;
  d.center_gap = d.call_lower - d.put_upper;
  d.max_profit_per_contract = d.butterfly_width * 100;

  // Update pin probability and confidence
  _updateDashboardMetrics(d);

  const ds = d.trade_date.replace(/-/g, '').substring(2);
  d.call_chain = `.SPXW${ds}C${Math.round(d.call_lower)}-2*.SPXW${ds}C${Math.round(d.call_center)}+.SPXW${ds}C${Math.round(d.call_upper)}`;
  d.put_chain = `.SPXW${ds}P${Math.round(d.put_upper)}-2*.SPXW${ds}P${Math.round(d.put_center)}+.SPXW${ds}P${Math.round(d.put_lower)}`;
  d.pnl_map = generatePnlMap(d);

  renderStrategy(d);
  saveState();
}

/* ── Rebuild centers symmetrically from open, gap, and width ── */
function _rebuildCenters(d) {
  // Round open to nearest strike FIRST to establish the anchor point.
  // Then build symmetrically from that anchor so both sides get
  // identical rounding treatment and the structure stays centered.
  const anchor = Math.round(d.spx_open / 5) * 5;
  const gap = d._inner_gap || 0;
  const halfGap = Math.round(gap / 2 / 5) * 5;  // snap half-gap to strike grid

  // Inner wings placed symmetrically around the anchor
  const callInner = anchor + halfGap;
  const putInner  = anchor - halfGap;

  // Centers are exactly one width away from the inner wings
  d.call_center = callInner + d.butterfly_width;
  d.put_center  = putInner  - d.butterfly_width;
}

/* ── Adjust butterfly width by ±5 — centers stay symmetric around open ── */
function adjustWidth(delta) {
  if (!strategyData) return;
  const d = strategyData;
  let newWidth = d.butterfly_width + delta;
  newWidth = Math.max(10, Math.round(newWidth / 5) * 5);
  d.butterfly_width = newWidth;
  _rebuildCenters(d);
  _recalcStrikes();
}

/* ── Adjust inner gap by ±10 — centers move symmetrically outward ── */
/* Steps by 10 so half-gap always lands on the 5-pt strike grid */
function adjustGap(delta) {
  if (!strategyData) return;
  const d = strategyData;
  const currentGap = d._inner_gap || 0;
  const newGap = Math.max(0, currentGap + delta);
  d._inner_gap = newGap;
  _rebuildCenters(d);
  _recalcStrikes();
}

function generatePnlMap(d) {
  const bw = d.butterfly_width;
  const margin = Math.max(bw * 1.5, d.expected_move * 0.5);
  const lo = d.put_lower - margin;
  const hi = d.call_upper + margin;
  const steps = 200;
  const results = [];
  for (let i = 0; i <= steps; i++) {
    const price = lo + (hi - lo) * i / steps;
    const call_pnl = Math.max(0, price - d.call_lower)
                   - 2 * Math.max(0, price - d.call_center)
                   + Math.max(0, price - d.call_upper);
    const put_pnl = Math.max(0, d.put_upper - price)
                  - 2 * Math.max(0, d.put_center - price)
                  + Math.max(0, d.put_lower - price);
    results.push({
      price: Math.round(price * 10) / 10,
      call_pnl: Math.round(call_pnl * 100) / 100,
      put_pnl: Math.round(put_pnl * 100) / 100,
      total_pnl: Math.round((call_pnl + put_pnl) * 100) / 100,
    });
  }
  return results;
}

/* ── Redraw P&L when debits change ── */
function redrawPnl() {
  if (!strategyData) return;
  const callDebit = parseFloat(document.getElementById('debitCall')?.value) || 0;
  const putDebit = parseFloat(document.getElementById('debitPut')?.value) || 0;

  // Check for trap debit (either call or put trap)
  let trapInfo = null;
  const trapCallDebit = parseFloat(document.getElementById('debitTrapCall')?.value) || 0;
  const trapPutDebit = parseFloat(document.getElementById('debitTrapPut')?.value) || 0;
  const d = strategyData;

  if (trapCallDebit > 0 && d.trap_call_center) {
    const tw = d.trap_width || Math.round(d.butterfly_width * 1.5 / 5) * 5;
    trapInfo = { center: d.trap_call_center, width: tw, debit: trapCallDebit };
  } else if (trapPutDebit > 0 && d.trap_put_center) {
    const tw = d.trap_width || Math.round(d.butterfly_width * 1.5 / 5) * 5;
    trapInfo = { center: d.trap_put_center, width: tw, debit: trapPutDebit };
  }

  drawPnlMap('pnlCanvas', d.pnl_map, d.spx_open,
             d.range_1sigma, callDebit, putDebit, d, trapInfo);
  updateNetDebit();
  saveState();
}

/* ── Main render ── */
function renderStrategy(d) {
  document.getElementById('results').classList.remove('hidden');

  // ── Confidence ──
  const conf = d.confidence_score;
  const confColor = conf >= 70 ? 'green' : conf >= 50 ? 'amber' : 'red';
  const confGrade = conf >= 70 ? 'high' : conf >= 50 ? 'med' : 'low';
  const profileColors = { conservative: 'var(--green)', moderate: 'var(--cyan)', aggressive: 'var(--amber)' };
  const profileColor = profileColors[d.risk_profile] || 'var(--cyan)';
  document.getElementById('confidenceGrid').innerHTML = `
    <div class="card text-center">
      <div class="metric ${confColor}"><div class="val" id="confValue">${conf}%</div><div class="label"><span class="tip" data-tip="Composite score based on width fit, pin probability, VIX regime, and gap penalty. Updates when you adjust width or gap.">Confidence</span></div></div>
      <div class="confidence-bar"><div class="fill confidence-${confGrade}" id="confBar" style="width:${conf}%"></div></div>
    </div>
    <div class="card text-center">
      <div class="metric green"><div class="val">±${d.expected_move}</div><div class="label"><span class="tip" data-tip="1σ daily move based on VIX. Width is sized to a risk-adjusted fraction of this value.">Expected Move (1σ)</span></div></div>
    </div>
    <div class="card text-center">
      <div class="metric cyan"><div class="val" id="pinValue">${d.pin_probability}%</div><div class="label"><span class="tip" data-tip="Probability SPX settles inside the butterfly profit zone. Uses fat-tailed distribution. Updates when you adjust width or gap.">Pin Probability</span></div></div>
    </div>
    <div class="card text-center">
      <div class="metric purple"><div class="val" id="maxProfitValue">$${d.max_profit_per_contract.toLocaleString()}</div><div class="label"><span class="tip" data-tip="Maximum profit if SPX settles exactly at the butterfly center strike. Equals width × $100.">Max / Contract</span></div></div>
    </div>
    ${d.risk_profile_desc ? `<div style="grid-column:1/-1;text-align:center;font-size:10px;color:${profileColor};padding:4px 0;letter-spacing:0.5px">${d.risk_profile_desc}</div>` : ''}
  `;

  // ── Parameters with editable width ──
  const regimeColors = { low_vol:'green', normal:'cyan', elevated:'amber', high_vol:'red' };
  const regimeColor = regimeColors[d.regime] || 'cyan';
  document.getElementById('paramsGrid').innerHTML = `
    <div class="card card-glow">
      <h3>Position Sizing</h3>
      <div class="grid-2" style="margin-bottom:16px">
        <div class="text-center">
          <div style="display:flex;align-items:center;justify-content:center;gap:8px">
            <button class="btn btn-secondary btn-sm" onclick="adjustWidth(-5)" style="padding:6px 12px;font-size:16px;font-weight:700">−</button>
            <div class="metric cyan" style="padding:4px 0"><div class="val" style="font-size:28px">${d.butterfly_width}</div><div class="label"><span class="tip" data-tip="Distance between strikes. Set to 60% of VIX-implied expected move. Wider = more profit potential but higher debit.">Butterfly Width</span></div></div>
            <button class="btn btn-secondary btn-sm" onclick="adjustWidth(5)" style="padding:6px 12px;font-size:16px;font-weight:700">+</button>
          </div>
        </div>
        <div class="text-center">
          <div style="display:flex;align-items:center;justify-content:center;gap:8px">
            <button class="btn btn-secondary btn-sm" onclick="adjustGap(-10)" style="padding:6px 12px;font-size:16px;font-weight:700">−</button>
            <div class="metric amber" style="padding:4px 0"><div class="val" style="font-size:28px">${d.center_gap}</div><div class="label"><span class="tip" data-tip="Space between the inner wings of the call and put flies. 0 = no dead zone (recommended). Larger gaps reduce debit but create a zone where both flies lose.">Inner Gap</span></div></div>
            <button class="btn btn-secondary btn-sm" onclick="adjustGap(10)" style="padding:6px 12px;font-size:16px;font-weight:700">+</button>
          </div>
        </div>
      </div>
      <div class="separator"></div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:16px">
        <div class="text-center"><div class="font-mono" style="font-size:15px;font-weight:600">${Math.round(d.call_center)}</div><div style="font-size:10px;color:var(--text-muted);margin-top:2px">CALL CTR</div></div>
        <div class="text-center"><div class="font-mono" style="font-size:15px;font-weight:600">${Math.round(d.put_center)}</div><div style="font-size:10px;color:var(--text-muted);margin-top:2px">PUT CTR</div></div>
        <div class="text-center"><div class="font-mono" style="font-size:15px;font-weight:600">${d.expected_move}</div><div style="font-size:10px;color:var(--text-muted);margin-top:2px">EXP MOVE</div></div>
        <div class="text-center"><div class="font-mono" style="font-size:15px;font-weight:600">${d.start_offset}</div><div style="font-size:10px;color:var(--text-muted);margin-top:2px">OFFSET</div></div>
      </div>
      <div style="font-size:11px;color:var(--text-muted);margin-top:12px;padding-top:12px;border-top:1px solid var(--border)">
        Strikes: <span class="font-mono">${Math.round(d.call_lower)}/${Math.round(d.call_center)}/${Math.round(d.call_upper)}</span> (C) &nbsp;
        <span class="font-mono">${Math.round(d.put_upper)}/${Math.round(d.put_center)}/${Math.round(d.put_lower)}</span> (P)
      </div>
    </div>
    <div class="card">
      <h3>Statistical Basis</h3>
      <table>
        <tr><td style="color:var(--text-dim);width:45%"><span class="tip" data-tip="Annualized VIX converted to daily. VIX 17 ≈ 1.07% daily σ.">Daily σ</span></td><td class="font-mono">${d.daily_sigma_pct}%</td></tr>
        <tr><td style="color:var(--text-dim)"><span class="tip" data-tip="Calibrated 1σ daily move in points. VIX overstatement and seasonality adjustments applied.">Expected Move</span></td><td class="font-mono">${d.expected_move} pts${d.raw_expected_move && d.raw_expected_move != d.expected_move ? ' <span style="color:var(--text-muted);font-size:10px">(VIX-implied: ' + d.raw_expected_move + ')</span>' : ''}</td></tr>
        <tr><td style="color:var(--text-dim)">1σ Range</td><td class="font-mono">${d.range_1sigma[0]} – ${d.range_1sigma[1]}</td></tr>
        <tr><td style="color:var(--text-dim)">2σ Range</td><td class="font-mono">${d.range_2sigma[0]} – ${d.range_2sigma[1]}</td></tr>
        <tr><td style="color:var(--text-dim)"><span class="tip" data-tip="VIX-based regime. Low (<14): tight flies. Normal (14-20): standard. Elevated (20-30): wide flies. High (30+): extra wide.">Regime</span></td><td><span class="tag tag-${regimeColor}">${d.regime.replace('_',' ')}</span></td></tr>
        <tr><td style="color:var(--text-dim)"><span class="tip" data-tip="1.5× the outer butterfly width. Wider trap = larger profit zone when trapping. Backtested optimal ratio.">Trap Width</span></td><td class="font-mono">${d.trap_width} pts</td></tr>
      </table>
    </div>
  `;

  // ── Chains ──
  document.getElementById('chainsGrid').innerHTML =
    chainCard('Call Butterfly', 'var(--green)', 'callChain', d.call_chain, 'debitCall') +
    chainCard('Put Butterfly', 'var(--red)', 'putChain', d.put_chain, 'debitPut');

  // ── Trap Builder ──
  const trapW = d.trap_width || Math.round(d.butterfly_width * 1.5 / 5) * 5;
  const triggerPts = d.trap_trigger_pts || Math.round(d.expected_move * 0.15);
  const callAlert = Math.round(d.spx_open + triggerPts);
  const putAlert = Math.round(d.spx_open - triggerPts);
  const tcc = d.trap_call_center || Math.round((d.spx_open + d.call_center) / 10) * 5;
  const tpc = d.trap_put_center || Math.round((d.spx_open + d.put_center) / 10) * 5;
  
  document.getElementById('trapInputCard').innerHTML = `
    <div style="padding:12px;background:var(--surface2);border-radius:8px;border-left:3px solid var(--purple);margin-bottom:12px">
      <div style="font-size:10px;font-weight:700;letter-spacing:1.5px;color:var(--purple);margin-bottom:8px">TRAP ALERTS <span class="tip" data-tip="When SPX moves ±${triggerPts}pts from open, buy a center butterfly to lock in profit from the directional fly. Close the losing side.">ⓘ</span></div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:13px">
        <div>
          <div style="color:var(--green);font-weight:700;font-size:11px;margin-bottom:4px">📈 CALL — SPX ≥ ${callAlert}</div>
          <div class="font-mono" style="font-size:14px">${tcc - trapW} / ${tcc} / ${tcc + trapW}</div>
        </div>
        <div>
          <div style="color:var(--red);font-weight:700;font-size:11px;margin-bottom:4px">📉 PUT — SPX ≤ ${putAlert}</div>
          <div class="font-mono" style="font-size:14px">${tpc - trapW} / ${tpc} / ${tpc + trapW}</div>
        </div>
      </div>
      <div style="margin-top:6px;font-size:10px;color:var(--text-muted)">
        Trigger: ±${triggerPts}pts (${(triggerPts/d.expected_move*100).toFixed(0)}% EM) • Optimal: 1:30–2:30 PM • Width: ${trapW}pt (1.5×)
      </div>
    </div>
    <div class="grid-3">
      <div><label>Trap Width (pts)</label><input type="number" id="trapWidth" value="${trapW}" step="5"></div>
      <div><label>Trigger Distance</label><div style="font-family:'JetBrains Mono',monospace;font-size:14px;padding:8px 0;color:var(--purple)">±${triggerPts}pts</div></div>
      <div style="display:flex;align-items:flex-end"><button class="btn btn-primary btn-block" onclick="calcTraps()">Generate Trap Chains</button></div>
    </div>
  `;

  // ── Trap Timeline — alert-based ──
  const recColors = { monitor: 'var(--text-dim)', cautious: 'var(--amber)', good: 'var(--cyan)', optimal: 'var(--green)', close: 'var(--red)', hedge: 'var(--purple)' };
  document.getElementById('trapTimeline').innerHTML = `
    <div style="font-size:11px;color:var(--text-dim);margin-bottom:10px;padding:8px 12px;background:var(--surface2);border-radius:6px;border-left:2px solid var(--purple)">
      <strong style="color:var(--purple)">Intraday Playbook</strong> —
      🟢 Trap (79% WR) &nbsp;|&nbsp; 🟣 Hedge on extended days (94% WR)
    </div>
    <table style="width:100%">
      <thead>
        <tr>
          <th>Time Window</th>
          <th>Action</th>
          <th style="text-align:center">θ Burned</th>
          <th style="text-align:right">Trap Debit</th>
        </tr>
      </thead>
      <tbody>
        ${d.trap_windows.map(w => `
          <tr style="border-bottom:1px solid var(--surface3)">
            <td style="padding:6px 10px">
              <span style="color:${recColors[w.recommendation] || 'var(--text)'};font-family:'JetBrains Mono',monospace;font-weight:600;font-size:12px">${w.label || w.time}</span>
              <div style="font-size:10px;color:var(--text-muted)">${w.time}</div>
            </td>
            <td style="padding:6px 10px;font-size:11px;color:${recColors[w.recommendation] || 'var(--text)'};font-weight:600">${(w.recommendation || '').toUpperCase()}</td>
            <td style="text-align:center;padding:6px 10px;font-family:'JetBrains Mono',monospace;font-size:12px">${w.theta_burned}%</td>
            <td style="text-align:right;padding:6px 10px;font-family:'JetBrains Mono',monospace;font-size:12px">${w.est_trap_debit != null ? '$' + w.est_trap_debit : '—'}</td>
          </tr>
          <tr><td colspan="4" style="padding:0 10px 8px 10px;font-size:10px;color:var(--text-muted);line-height:1.4">${w.note || ''}</td></tr>
        `).join('')}
      </tbody>
    </table>
  `;

  // ── Charts — wait for layout to complete ──
  requestAnimationFrame(() => {
    setTimeout(() => {
      drawThetaCurve('thetaCanvas', d.theta_curve_data);
      const callDebit = parseFloat(document.getElementById('debitCall')?.value) || 0;
      const putDebit = parseFloat(document.getElementById('debitPut')?.value) || 0;
      drawPnlMap('pnlCanvas', d.pnl_map, d.spx_open, d.range_1sigma, callDebit, putDebit, d, null);
    }, 50);
  });

  // Live-refresh theta chart every 60s to move the NOW marker
  if (window._thetaRefreshInterval) clearInterval(window._thetaRefreshInterval);
  window._thetaRefreshInterval = setInterval(() => {
    if (d.theta_curve_data) drawThetaCurve('thetaCanvas', d.theta_curve_data);
  }, 60000);

  // ── Summary ──
  const regimeLabel = d.regime.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
  const trapLow = Math.round(d.put_upper || 0);
  const trapHigh = Math.round(d.call_lower || 0);
  const maxLoss = d.butterfly_width * 2; // debit on both sides, rough estimate
  document.getElementById('summaryCard').innerHTML = `
    <div style="display:grid;grid-template-columns:1fr 1px 1fr;gap:0 24px">
      <div>
        <div style="font-size:10px;font-weight:700;letter-spacing:1.5px;color:var(--accent);margin-bottom:10px">STEP 1 — OPEN (9:30 AM)</div>
        <div style="display:grid;grid-template-columns:auto 1fr;gap:4px 10px;font-size:13px;align-items:center">
          <span style="color:var(--green);font-weight:700;font-size:11px">CALL</span>
          <span class="font-mono">${Math.round(d.call_lower)} / ${Math.round(d.call_center)} / ${Math.round(d.call_upper)}</span>
          <span style="color:var(--red);font-weight:700;font-size:11px">PUT</span>
          <span class="font-mono">${Math.round(d.put_upper)} / ${Math.round(d.put_center)} / ${Math.round(d.put_lower)}</span>
        </div>
        <div style="margin-top:10px;font-size:11px;color:var(--text-muted)">
          ${d.butterfly_width}pt wide • ${d.center_gap}pt gap • Max $${d.max_profit_per_contract}/ct
        </div>
      </div>
      <div style="background:var(--border)"></div>
      <div>
        <div style="font-size:10px;font-weight:700;letter-spacing:1.5px;color:var(--amber);margin-bottom:10px">STEP 2 — TRAP (1:30–2:30 PM)</div>
        <div style="font-size:12px;line-height:1.8;color:var(--text-dim)">
          <div><span style="color:var(--green)">📈 SPX ≥ ${callAlert}</span> → <span class="font-mono">${tcc-trapW}/${tcc}/${tcc+trapW}</span></div>
          <div><span style="color:var(--red)">📉 SPX ≤ ${putAlert}</span> → <span class="font-mono">${tpc-trapW}/${tpc}/${tpc+trapW}</span></div>
          <div style="color:var(--text-muted);font-size:11px;margin-top:4px">Close losing side • Hold to expiry</div>
        </div>
      </div>
    </div>
  `;

  document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ── Trap Calculator ── */
async function calcTraps() {
  if (!strategyData) return;
  const d = strategyData;
  try {
    const resp = await fetch('/api/traps', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        trade_date: d.trade_date, spx_open: d.spx_open,
        call_center: d.call_center, put_center: d.put_center,
        butterfly_width: d.butterfly_width,
        trap_width: parseFloat(document.getElementById('trapWidth').value),
      }),
    });
    if (resp.status === 401) { window.location.href = '/login?next=/batman'; return; }
    if (resp.status === 403) { window.location.href = '/account'; return; }
    const json = await resp.json();
    if (json.status !== 'ok') throw new Error(json.message);
    const chains = json.data;
    const el = document.getElementById('trapResults');
    el.classList.remove('hidden');
    el.innerHTML = `
      <div style="font-size:12px;color:var(--text-dim);margin-bottom:12px;padding:8px 12px;background:var(--surface2);border-radius:8px;border-left:3px solid var(--accent)">
        Trap centered at <strong style="color:var(--accent)">${chains.trap_center}</strong> (open), width <strong>${chains.trap_width}</strong>pts
      </div>
      <div class="grid-2">
        ${chainCard('Trap Call Fly', 'var(--green)', 'trapCallChain', chains.trap_call_fly, 'debitTrapCall')}
        ${chainCard('Trap Put Fly', 'var(--red)', 'trapPutChain', chains.trap_put_fly, 'debitTrapPut')}
      </div>
      <div class="grid-2 mt-3">
        ${chainCard('Outside Call', 'var(--text-dim)', 'outCallChain', chains.outside_call, 'debitOutCall')}
        ${chainCard('Outside Put', 'var(--text-dim)', 'outPutChain', chains.outside_put, 'debitOutPut')}
      </div>
    `;
  } catch (err) { showToast('Trap error: ' + err.message, 'error'); }
}

/* ── Net Debit ── */
function updateNetDebit() {
  let total = 0;
  document.querySelectorAll('.debit-input').forEach(el => { total += parseFloat(el.value) || 0; });
  document.getElementById('netDebit').textContent = total.toFixed(2);
}
