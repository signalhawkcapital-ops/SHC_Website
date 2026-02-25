/* ── directional.js — Directional Signals Page ── */

async function runDirectional() {
  const btn = document.getElementById('dRunBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="pulse">Analyzing...</span>';
  document.getElementById('dLoading').classList.remove('hidden');

  try {
    const resp = await fetch('/api/directional', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        spx_open: parseFloat(document.getElementById('dSpxOpen').value),
        vix: parseFloat(document.getElementById('dVix').value),
        trade_date: document.getElementById('dDate').value,
      }),
    });
    const json = await resp.json();
    if (json.status !== 'ok') throw new Error(json.message);
    renderDirectional(json.data);
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '⚡ Generate Signals';
    document.getElementById('dLoading').classList.add('hidden');
  }
}

function renderDirectional(d) {
  document.getElementById('dResults').classList.remove('hidden');

  const dirColor = d.consensus_direction === 'CALL' ? 'green' :
                   d.consensus_direction === 'PUT' ? 'red' : 'text-dim';
  const dirIcon = d.consensus_direction === 'CALL' ? '▲' :
                  d.consensus_direction === 'PUT' ? '▼' : '—';

  // Consensus
  document.getElementById('consensusGrid').innerHTML = `
    <div class="card text-center">
      <div class="metric ${dirColor}">
        <div class="val">${dirIcon} ${d.consensus_direction}</div>
        <div class="label">Consensus</div>
      </div>
    </div>
    <div class="card text-center">
      <div class="metric ${d.consensus_confidence > 60 ? 'green' : d.consensus_confidence > 40 ? 'amber' : 'red'}">
        <div class="val">${d.consensus_confidence}%</div>
        <div class="label">Confidence</div>
      </div>
    </div>
    <div class="card text-center">
      <div class="metric cyan">
        <div class="val">${d.recommended_strike}</div>
        <div class="label">Rec. Strike</div>
      </div>
    </div>
    <div class="card text-center">
      <div class="metric purple">
        <div class="val">${d.recommended_entry}</div>
        <div class="label">Entry Time</div>
      </div>
    </div>
  `;

  // Signal cards
  const stratColors = { 'ORB': 'cyan', 'MR Fade': 'purple', 'VWAP Mom': 'green', 'Gap Play': 'amber' };
  const stratIcons = { 'ORB': '📊', 'MR Fade': '↩', 'VWAP Mom': '📈', 'Gap Play': '🔀' };

  document.getElementById('signalCards').innerHTML = d.signals.map(s => {
    const color = stratColors[s.strategy] || 'cyan';
    const icon = stratIcons[s.strategy] || '•';
    const sigDir = s.direction === 'CALL' ? 'green' : s.direction === 'PUT' ? 'red' : 'text-dim';
    const sigIcon = s.direction === 'CALL' ? '▲ CALL' : s.direction === 'PUT' ? '▼ PUT' : '— NONE';

    const conditions = [
      ...s.conditions_met.map(c => `<div style="color:var(--green);font-size:12px;margin:3px 0">✓ ${c}</div>`),
      ...s.conditions_failed.map(c => `<div style="color:var(--red);font-size:12px;margin:3px 0">✗ ${c}</div>`),
    ].join('');

    return `
      <div class="card" style="border-left:3px solid var(--${color})">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
          <h3 style="color:var(--${color});margin:0">${icon} ${s.strategy}</h3>
          <span class="tag tag-${sigDir === 'green' ? 'green' : sigDir === 'red' ? 'red' : 'cyan'}" style="font-size:13px;font-weight:700">${sigIcon}</span>
        </div>
        <div style="display:flex;gap:20px;margin-bottom:12px">
          <div class="text-center">
            <div class="font-mono" style="font-size:20px;font-weight:700;color:var(--${s.confidence > 55 ? 'green' : s.confidence > 40 ? 'amber' : 'red'})">${s.confidence}%</div>
            <div style="font-size:10px;color:var(--text-muted)">CONFIDENCE</div>
          </div>
          <div class="text-center">
            <div class="font-mono" style="font-size:16px;color:var(--text)">${s.entry_time}</div>
            <div style="font-size:10px;color:var(--text-muted)">ENTRY</div>
          </div>
          <div class="text-center">
            <div class="font-mono" style="font-size:16px;color:var(--text)">${s.stop_pct}/${s.target_pct}%</div>
            <div style="font-size:10px;color:var(--text-muted)">STOP/TARGET</div>
          </div>
        </div>
        <div style="font-size:13px;color:var(--text-dim);margin-bottom:8px">${s.rationale}</div>
        <div style="border-top:1px solid var(--border);padding-top:8px;margin-top:8px">${conditions}</div>
      </div>`;
  }).join('');

  // Trade Rec
  const bestSignal = d.signals.filter(s => s.direction !== 'NONE').sort((a, b) => b.confidence - a.confidence)[0];
  if (bestSignal && d.consensus_direction !== 'NEUTRAL') {
    const recDir = d.consensus_direction;
    const recColor = recDir === 'CALL' ? 'green' : 'red';
    const strike = d.recommended_strike;
    const dateStr = d.date.replace(/-/g, '').substring(2);
    const optType = recDir === 'CALL' ? 'C' : 'P';
    const chain = `.SPXW${dateStr}${optType}${strike}`;

    document.getElementById('tradeRec').innerHTML = `
      <div class="grid-2">
        <div>
          <h3 style="color:var(--${recColor})">Recommended: Buy ${recDir}</h3>
          <div style="margin:8px 0">
            <div class="chain-output" id="recChain">${chain}</div>
            <button class="btn btn-secondary btn-sm copy-btn mt-2" onclick="copyText('recChain', this)">Copy Chain</button>
          </div>
          <table style="margin-top:12px">
            <tr><td style="color:var(--text-dim)">Strike</td><td class="font-mono">${strike} (${Math.abs(bestSignal.strike_offset).toFixed(0)}pt ${recDir === 'CALL' ? 'OTM' : 'OTM'})</td></tr>
            <tr><td style="color:var(--text-dim)">Entry</td><td class="font-mono">${bestSignal.entry_time} ET</td></tr>
            <tr><td style="color:var(--text-dim)">Stop</td><td class="font-mono">${bestSignal.stop_pct}% of premium</td></tr>
            <tr><td style="color:var(--text-dim)">Target</td><td class="font-mono">${bestSignal.target_pct}% of premium</td></tr>
          </table>
        </div>
        <div>
          <h3>Signal Confluence</h3>
          <div style="margin:8px 0">
            ${d.signals.map(s => {
              const agree = s.direction === recDir;
              const color = s.direction === 'NONE' ? 'var(--text-muted)' : agree ? 'var(--green)' : 'var(--red)';
              const icon = s.direction === 'NONE' ? '—' : agree ? '✓' : '✗';
              return `<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--surface3)">
                <span style="color:var(--text-dim)">${s.strategy}</span>
                <span style="color:${color};font-weight:600">${icon} ${s.direction} (${s.confidence}%)</span>
              </div>`;
            }).join('')}
          </div>
          <div style="font-size:12px;color:var(--text-muted);margin-top:12px">
            ${d.risk_reward}
          </div>
        </div>
      </div>`;
  } else {
    document.getElementById('tradeRec').innerHTML = `
      <div class="text-center" style="padding:20px">
        <div style="font-size:20px;color:var(--text-muted)">— No Clear Signal —</div>
        <p style="color:var(--text-dim);margin-top:8px">Strategies are conflicting or insufficient confidence. Consider sitting out today.</p>
      </div>`;
  }

  document.getElementById('dResults').scrollIntoView({ behavior: 'smooth' });
}

/* ── Copy ── */
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

/* ── Directional Backtest ── */
async function runDirBacktest() {
  const btn = document.getElementById('dBtBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="pulse">Backtesting...</span>';
  document.getElementById('dBtLoading').classList.remove('hidden');

  try {
    const resp = await fetch('/api/directional/backtest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        start_date: document.getElementById('dBtStart').value,
        end_date: document.getElementById('dBtEnd').value,
      }),
    });
    const json = await resp.json();
    if (json.status !== 'ok') throw new Error(json.message);
    renderDirBacktest(json.data);
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '▶ Run Backtest';
    document.getElementById('dBtLoading').classList.add('hidden');
  }
}

function renderDirBacktest(results) {
  document.getElementById('dBtResults').classList.remove('hidden');

  const colorMap = { 'ORB': 'cyan', 'MR Fade': 'purple', 'VWAP Mom': 'green', 'Gap Play': 'amber' };

  document.getElementById('btStratCards').innerHTML = results.map(r => {
    const color = colorMap[r.strategy] || 'cyan';
    const wrColor = r.win_rate > 55 ? 'green' : r.win_rate > 48 ? 'amber' : 'red';
    const pfColor = r.profit_factor > 1.2 ? 'green' : r.profit_factor > 1.0 ? 'amber' : 'red';

    return `
      <div class="card" style="border-left:3px solid var(--${color})">
        <h3 style="color:var(--${color});margin-bottom:16px">${r.strategy}</h3>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:16px">
          <div class="text-center">
            <div class="font-mono" style="font-size:22px;font-weight:700;color:var(--${wrColor})">${r.win_rate}%</div>
            <div style="font-size:10px;color:var(--text-muted)">WIN RATE</div>
          </div>
          <div class="text-center">
            <div class="font-mono" style="font-size:22px;font-weight:700;color:var(--${pfColor})">${r.profit_factor}</div>
            <div style="font-size:10px;color:var(--text-muted)">PROFIT FACTOR</div>
          </div>
          <div class="text-center">
            <div class="font-mono" style="font-size:22px;font-weight:700">${r.total_signals}</div>
            <div style="font-size:10px;color:var(--text-muted)">SIGNALS</div>
          </div>
        </div>
        <table style="font-size:12px">
          <tr><td style="color:var(--text-dim)">Correct / Wrong</td><td class="font-mono">${r.correct} / ${r.wrong}</td></tr>
          <tr><td style="color:var(--text-dim)">No Signal Days</td><td class="font-mono">${r.no_signal}</td></tr>
          <tr><td style="color:var(--text-dim)">Avg Move (correct)</td><td class="font-mono" style="color:var(--green)">${r.avg_move_when_correct} pts</td></tr>
          <tr><td style="color:var(--text-dim)">Avg Move (wrong)</td><td class="font-mono" style="color:var(--red)">${r.avg_move_when_wrong} pts</td></tr>
          <tr><td style="color:var(--text-dim)">Avg R:R (pts)</td><td class="font-mono">${r.avg_rr_pts > 0 ? '+' : ''}${r.avg_rr_pts}</td></tr>
          <tr><td style="color:var(--text-dim)">Best Regime</td><td><span class="tag tag-green">${r.best_regime}</span></td></tr>
          <tr><td style="color:var(--text-dim)">Best Day</td><td><span class="tag tag-cyan">${r.best_dow}</span></td></tr>
        </table>
        <div style="border-top:1px solid var(--border);padding-top:10px;margin-top:10px">
          ${r.notes.map(n => `<div style="font-size:12px;margin:3px 0">${n}</div>`).join('')}
        </div>
      </div>`;
  }).join('');

  // Insights
  const best = results.reduce((a, b) => a.profit_factor > b.profit_factor ? a : b);
  const profitable = results.filter(r => r.profit_factor > 1.0);
  document.getElementById('btInsights').innerHTML = `
    <div style="font-size:14px;line-height:1.8">
      <strong style="color:var(--accent)">Summary:</strong>
      ${profitable.length} of ${results.length} strategies are net profitable over the tested period.<br>
      <strong>Best strategy:</strong> <span style="color:var(--green)">${best.strategy}</span> with ${best.win_rate}% win rate and ${best.profit_factor} profit factor.<br>
      ${profitable.length >= 3 ? '✅ Multiple confirming strategies increase confidence when they agree.' : '⚠ Limited edge — use strict risk management and position sizing.'}
    </div>`;

  document.getElementById('dBtResults').scrollIntoView({ behavior: 'smooth' });
}
