/* ── backtest.js — Backtest dashboard logic ── */

async function runBacktest() {
  const btn = document.getElementById('btRunBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Running…';
  document.getElementById('btLoading').classList.remove('hidden');
  document.getElementById('btResults').classList.add('hidden');
  document.getElementById('optResults').classList.add('hidden');

  const payload = {
    data_source: document.getElementById('dataSource').value,
    start_date: document.getElementById('btStart').value,
    end_date: document.getElementById('btEnd').value,
    risk_profile: document.getElementById('btRisk').value,
    min_vix: parseFloat(document.getElementById('btMinVix').value),
    max_vix: parseFloat(document.getElementById('btMaxVix').value),
    debit_per_side: document.getElementById('btDebit').value || null,
  };
  // Only include synthetic params if synthetic source
  if (payload.data_source === 'synthetic') {
    payload.initial_spx = parseFloat(document.getElementById('btSpx').value);
    payload.initial_vix = parseFloat(document.getElementById('btVix').value);
    payload.seed = 42;
  }

  try {
    const resp = await fetch('/api/backtest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const json = await resp.json();
    if (json.status !== 'ok') throw new Error(json.message);
    renderBacktest(json.data);
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = '▶ Run Backtest';
    document.getElementById('btLoading').classList.add('hidden');
  }
}

async function runOptimize() {
  const btn = document.getElementById('optRunBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Optimizing…';
  document.getElementById('btLoading').classList.remove('hidden');
  document.getElementById('optResults').classList.add('hidden');

  const payload = {
    data_source: document.getElementById('dataSource').value,
    start_date: document.getElementById('btStart').value,
    end_date: document.getElementById('btEnd').value,
    risk_profile: document.getElementById('btRisk').value,
  };
  if (payload.data_source === 'synthetic') {
    payload.initial_spx = parseFloat(document.getElementById('btSpx').value);
    payload.initial_vix = parseFloat(document.getElementById('btVix').value);
    payload.seed = 42;
  }

  try {
    const resp = await fetch('/api/optimize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const json = await resp.json();
    if (json.status !== 'ok') throw new Error(json.message);
    renderOptimize(json.data);
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = '🔍 Optimize Parameters';
    document.getElementById('btLoading').classList.add('hidden');
  }
}

function renderBacktest(d) {
  document.getElementById('btResults').classList.remove('hidden');

  // ── Data source context banner ──
  let contextHtml = '';
  const isReal = d._data_source === 'real';
  if (isReal && d._date_range) {
    const start = d._date_range[0], end = d._date_range[1];
    const spxStart = d._spx_start ? d._spx_start.toFixed(0) : '?';
    const spxEnd = d._spx_end ? d._spx_end.toFixed(0) : '?';
    const pnlDollars = (d.total_pnl * 100).toLocaleString('en', {maximumFractionDigits:0});
    const daily = d.avg_pnl > 0 ? `+$${(d.avg_pnl * 100).toFixed(0)}` : `-$${Math.abs(d.avg_pnl * 100).toFixed(0)}`;
    contextHtml = `
      <div class="card mb-3" style="border-color:var(--green);background:linear-gradient(135deg,var(--green-dim),var(--surface))">
        <div style="font-size:10px;font-weight:700;letter-spacing:1.5px;color:var(--green);margin-bottom:6px">REAL SPX/VIX DATA — HYPOTHETICAL BACKTEST</div>
        <div style="font-size:14px;color:var(--text);line-height:1.5">
          Wingspan strategy applied to <strong>every trading day ${start} – ${end}</strong>
          (SPX ${spxStart} → ${spxEnd}): hypothetical result of
          <strong style="color:${d.total_pnl >= 0 ? 'var(--green)' : 'var(--red)'}">$${pnlDollars}/contract</strong>
          over ${d.total_days} days (${daily}/day).
        </div>
        <div style="margin-top:8px;font-size:10px;color:var(--text-muted);line-height:1.4;padding-top:8px;border-top:1px solid var(--border)">
          Does not account for slippage, commissions (~$4–$8/day), partial fills, or spread widening. Est. net after costs: <strong style="color:var(--text-dim)">$${Math.max(0, (d.avg_pnl * 100) - 8).toFixed(0)}–$${Math.max(0, (d.avg_pnl * 100) - 4).toFixed(0)}/day</strong>.
          <a href="/disclosures" style="color:var(--accent)">Risk Disclosures →</a>
        </div>
      </div>`;
  } else if (d._data_source === 'synthetic') {
    contextHtml = `
      <div class="card mb-3" style="border-color:var(--amber);padding:10px 16px">
        <div style="font-size:10px;font-weight:700;letter-spacing:1.5px;color:var(--amber);margin-bottom:2px">SYNTHETIC DATA — HYPOTHETICAL</div>
        <div style="font-size:11px;color:var(--text-dim)">Stochastic vol model, not real prices. Select "Real SPX/VIX" for actual market data.</div>
      </div>`;
  }

  // Performance cards
  const pnlColor = d.total_pnl >= 0 ? 'green' : 'red';
  document.getElementById('perfCards').innerHTML = contextHtml + `
    <div class="grid-4">
    <div class="card"><div class="metric ${pnlColor}"><div class="val">${d.total_pnl >= 0 ? '+' : ''}${d.total_pnl.toFixed(1)}</div><div class="label">Total P&L (pts) · $${(d.total_pnl * 100).toLocaleString('en', {maximumFractionDigits:0})}/ct</div></div></div>
    <div class="card"><div class="metric ${d.win_rate >= 50 ? 'green' : 'amber'}"><div class="val">${d.win_rate}%</div><div class="label">Win Rate (${d.total_days} days)</div></div></div>
    <div class="card"><div class="metric cyan"><div class="val">${d.sharpe_ratio}</div><div class="label">Sharpe Ratio</div></div></div>
    <div class="card"><div class="metric purple"><div class="val">${d.profit_factor}</div><div class="label">Profit Factor</div></div></div>
    </div>`;

  // Additional metrics row
  const extra = `
    <div class="grid-4 mt-3">
      <div class="card"><div class="metric"><div class="val" style="font-size:20px;color:var(--text)">${d.avg_pnl.toFixed(2)}</div><div class="label">Avg P&L</div></div></div>
      <div class="card"><div class="metric green"><div class="val" style="font-size:20px">${d.avg_winner.toFixed(2)}</div><div class="label">Avg Winner</div></div></div>
      <div class="card"><div class="metric red"><div class="val" style="font-size:20px">${d.avg_loser.toFixed(2)}</div><div class="label">Avg Loser</div></div></div>
      <div class="card"><div class="metric amber"><div class="val" style="font-size:20px">${d.max_drawdown.toFixed(1)}</div><div class="label">Max Drawdown</div></div></div>
    </div>
    <div class="grid-4 mt-3">
      <div class="card"><div class="metric"><div class="val" style="font-size:20px;color:var(--text)">${d.max_consecutive_losses}</div><div class="label">Max Consec Losses</div></div></div>
      <div class="card"><div class="metric cyan"><div class="val" style="font-size:20px">${(d.pct_1sigma_days || 0).toFixed(0)}%</div><div class="label">Days Within 1σ</div></div></div>
      <div class="card"><div class="metric ${d.trap_vs_hold_improvement > 0 ? 'green' : 'red'}"><div class="val" style="font-size:20px">${d.trap_vs_hold_improvement > 0 ? '+' : ''}${(d.trap_vs_hold_improvement || 0).toFixed(2)}</div><div class="label">Trap vs Hold Δ</div></div></div>
      <div class="card"><div class="metric purple"><div class="val" style="font-size:20px">${(d.trap_hit_rate || 0).toFixed(0)}%</div><div class="label">Trap Hit Rate</div></div></div>
    </div>
    <div class="grid-4 mt-3">
      <div class="card"><div class="metric ${d.hedge_rate > 0 ? 'purple' : ''}"><div class="val" style="font-size:20px">${(d.hedge_rate || 0).toFixed(0)}%</div><div class="label">Hedge Rate</div></div></div>
      <div class="card"><div class="metric ${d.hedge_win_rate > 50 ? 'green' : 'amber'}"><div class="val" style="font-size:20px">${(d.hedge_win_rate || 0).toFixed(0)}%</div><div class="label">Hedge Win Rate</div></div></div>
      <div class="card"><div class="metric ${d.avg_hedge_pnl >= 0 ? 'green' : 'red'}"><div class="val" style="font-size:20px">${d.avg_hedge_pnl >= 0 ? '+' : ''}${(d.avg_hedge_pnl || 0).toFixed(2)}</div><div class="label">Avg Hedge P&L</div></div></div>
      <div class="card"><div class="metric"><div class="val" style="font-size:20px;color:var(--text)">${(d.avg_butterfly_width || 0).toFixed(0)}pt</div><div class="label">Avg Fly Width</div></div></div>
    </div>
    <div class="card mt-3" style="border-color:var(--amber)">
      <h3 style="color:var(--amber)">Estimated Real-World Costs <span class="tip" data-tip="These deductions are estimates based on typical retail broker pricing and average 0DTE spread behavior. Actual costs vary by broker, time of day, and market conditions.">ⓘ</span></h3>
      <div class="grid-4 gap-3" style="font-size:12px">
        <div class="text-center">
          <div style="color:var(--amber);font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:600">~$6</div>
          <div style="color:var(--text-muted);font-size:10px;margin-top:2px">EST. COMMISSIONS/DAY</div>
          <div style="color:var(--text-muted);font-size:9px">$0.65/leg × ~9 legs avg</div>
        </div>
        <div class="text-center">
          <div style="color:var(--amber);font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:600">~$20</div>
          <div style="color:var(--text-muted);font-size:10px;margin-top:2px">EST. SLIPPAGE/DAY</div>
          <div style="color:var(--text-muted);font-size:9px">$0.05–$0.15 per leg × 2 sides</div>
        </div>
        <div class="text-center">
          <div style="color:var(--text);font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:600">$${Math.max(0, (d.avg_pnl * 100) - 26).toFixed(0)}</div>
          <div style="color:var(--text-muted);font-size:10px;margin-top:2px">NET AVG P&L/DAY</div>
          <div style="color:var(--text-muted);font-size:9px">After est. costs ($${(d.avg_pnl * 100).toFixed(0)} − $26)</div>
        </div>
        <div class="text-center">
          <div style="color:${(d.avg_pnl * 100 - 26) * d.total_days >= 0 ? 'var(--green)' : 'var(--red)'};font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:600">$${((d.avg_pnl * 100 - 26) * d.total_days).toLocaleString('en', {maximumFractionDigits:0})}</div>
          <div style="color:var(--text-muted);font-size:10px;margin-top:2px">NET TOTAL (EST.)</div>
          <div style="color:var(--text-muted);font-size:9px">${d.total_days} days after costs</div>
        </div>
      </div>
    </div>
  `;
  document.getElementById('perfCards').insertAdjacentHTML('afterend', extra);

  // Charts
  setTimeout(() => {
    drawEquityCurve('equityCanvas', d.trades);
    drawHistogram('histCanvas', d.trades.map(t => t.net_pnl));
    drawMonthlyBars('monthlyCanvas', d.monthly_stats);

    // Outcome donut
    drawOutcomeChart('outcomeCanvas', d.trades);
  }, 200);

  // Regime table
  const regimeHtml = `<table><thead><tr><th>Regime</th><th>Trades</th><th>Win Rate</th><th>Avg P&L</th><th>Total P&L</th></tr></thead><tbody>
    ${Object.entries(d.regime_stats).map(([k, v]) => `<tr>
      <td><span class="tag tag-cyan">${k.replace('_',' ')}</span></td>
      <td>${v.count}</td>
      <td style="color:${v.win_rate >= 50 ? 'var(--green)' : 'var(--red)'}">${v.win_rate}%</td>
      <td style="color:${v.avg_pnl >= 0 ? 'var(--green)' : 'var(--red)'}">$${v.avg_pnl}</td>
      <td style="color:${v.total_pnl >= 0 ? 'var(--green)' : 'var(--red)'}">$${v.total_pnl}</td>
    </tr>`).join('')}</tbody></table>`;
  document.getElementById('regimeTable').innerHTML = regimeHtml;

  // DOW table
  const dowHtml = `<table><thead><tr><th>Day</th><th>Trades</th><th>Win Rate</th><th>Avg P&L</th><th>Total</th></tr></thead><tbody>
    ${Object.entries(d.dow_stats).sort(([a],[b]) => a-b).map(([k, v]) => `<tr>
      <td>${v.day_name}</td>
      <td>${v.count}</td>
      <td style="color:${v.win_rate >= 50 ? 'var(--green)' : 'var(--red)'}">${v.win_rate}%</td>
      <td style="color:${v.avg_pnl >= 0 ? 'var(--green)' : 'var(--red)'}">$${v.avg_pnl}</td>
      <td style="color:${v.total_pnl >= 0 ? 'var(--green)' : 'var(--red)'}">$${v.total_pnl}</td>
    </tr>`).join('')}</tbody></table>`;
  document.getElementById('dowTable').innerHTML = dowHtml;

  // Calibration
  document.getElementById('calibrationCard').innerHTML = d.calibration_notes.map(n =>
    `<p style="margin:8px 0;padding:10px 14px;background:var(--surface2);border-radius:8px;font-size:13px;color:var(--text-dim)">${n}</p>`
  ).join('');

  // ── Trap Analysis Dashboard ──
  buildTrapAnalysis(d);

  // Trade log (last 50)
  const lastTrades = d.trades.slice(-50).reverse();
  const logHead = `<tr><th>Date</th><th>Open</th><th>Close</th><th>VIX</th><th>Width</th>
    <th>Regime</th><th>Hold P&L</th><th>Outcome</th>
    <th>Trap Time</th><th>Trap ₵</th><th>Trap Debit</th><th>Combined</th><th>Δ</th></tr>`;
  const logBody = lastTrades.map(t => {
    const outcomeTag = {
      max_profit: 'tag-green', partial_profit: 'tag-cyan',
      breakeven: 'tag-amber', loss: 'tag-red', full_loss: 'tag-red'
    }[t.outcome] || 'tag-amber';
    const trapped = t.could_have_trapped;
    return `<tr>
      <td>${t.trade_date}</td><td>${t.spx_open.toFixed(0)}</td><td>${t.spx_close.toFixed(0)}</td>
      <td>${t.vix.toFixed(1)}</td><td>${t.butterfly_width}</td>
      <td><span class="tag tag-cyan" style="font-size:10px">${t.regime.replace('_',' ')}</span></td>
      <td style="color:${t.net_pnl >= 0 ? 'var(--green)' : 'var(--red)'}">${t.net_pnl >= 0 ? '+' : ''}$${t.net_pnl.toFixed(2)}</td>
      <td><span class="tag ${outcomeTag}" style="font-size:10px">${t.outcome.replace('_',' ')}</span></td>
      <td>${trapped ? t.optimal_trap_time : '<span style="color:var(--text-muted)">—</span>'}</td>
      <td>${trapped ? t.trap_center : '—'}</td>
      <td>${trapped ? '$' + t.trap_debit.toFixed(2) : '—'}</td>
      <td style="color:${trapped && t.trapped_pnl >= 0 ? 'var(--green)' : trapped ? 'var(--red)' : 'var(--text-muted)'}">
        ${trapped ? (t.trapped_pnl >= 0 ? '+' : '') + '$' + t.trapped_pnl.toFixed(2) : '—'}</td>
      <td style="color:${t.trap_improvement > 0 ? 'var(--green)' : t.trap_improvement < 0 ? 'var(--red)' : 'var(--text-muted)'}">
        ${trapped ? (t.trap_improvement > 0 ? '+' : '') + '$' + t.trap_improvement.toFixed(2) : ''}</td>
    </tr>`;
  }).join('');
  document.querySelector('#tradeLog thead').innerHTML = logHead;
  document.querySelector('#tradeLog tbody').innerHTML = logBody;

  document.getElementById('btResults').scrollIntoView({ behavior: 'smooth' });
}

function drawOutcomeChart(canvasId, trades) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const { ctx, w, h } = setupCanvas(canvas);

  const outcomes = {};
  trades.forEach(t => { outcomes[t.outcome] = (outcomes[t.outcome] || 0) + 1; });

  const total = trades.length;
  const items = Object.entries(outcomes).sort(([,a],[,b]) => b - a);
  const colors = {
    max_profit: '#22c55e', partial_profit: '#06b6d4',
    breakeven: '#f59e0b', loss: '#ef4444', blown: '#dc2626'
  };

  const cx = w / 2, cy = h / 2 - 10, r = Math.min(w, h) / 2 - 40;
  let startAngle = -Math.PI / 2;

  items.forEach(([key, count]) => {
    const slice = (count / total) * 2 * Math.PI;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, r, startAngle, startAngle + slice);
    ctx.closePath();
    ctx.fillStyle = colors[key] || '#4f6ef7';
    ctx.fill();
    startAngle += slice;
  });

  // Inner circle (donut)
  ctx.beginPath();
  ctx.arc(cx, cy, r * 0.55, 0, Math.PI * 2);
  ctx.fillStyle = '#12131a';
  ctx.fill();

  // Center text
  ctx.fillStyle = '#e8e9f0';
  ctx.font = '24px JetBrains Mono';
  ctx.textAlign = 'center';
  ctx.fillText(total, cx, cy + 8);
  ctx.font = '10px Sora';
  ctx.fillStyle = '#8a8b9a';
  ctx.fillText('TRADES', cx, cy + 24);

  // Legend
  let ly = h - 20;
  ctx.font = '11px Sora';
  ctx.textAlign = 'left';
  items.reverse().forEach(([key, count]) => {
    const pct = ((count / total) * 100).toFixed(0);
    ctx.fillStyle = colors[key] || '#4f6ef7';
    ctx.fillRect(10, ly - 8, 10, 10);
    ctx.fillStyle = '#8a8b9a';
    ctx.fillText(`${key.replace('_',' ')} (${pct}%)`, 26, ly);
    ly -= 16;
  });
}

function renderOptimize(d) {
  document.getElementById('optResults').classList.remove('hidden');

  document.getElementById('optCards').innerHTML = `
    <div class="card"><div class="metric green"><div class="val">${d.best_sharpe}</div><div class="label">Best Sharpe</div></div></div>
    <div class="card"><div class="metric cyan"><div class="val">${d.best_width_pct}</div><div class="label">Best Width %</div></div></div>
    <div class="card"><div class="metric amber"><div class="val">${d.best_gap_pct}</div><div class="label">Best Gap %</div></div></div>
  `;

  document.getElementById('optNotes').innerHTML = `<h3>Optimization Notes</h3>` +
    d.notes.map(n => `<p style="margin:8px 0;color:var(--text-dim);font-size:13px">${n}</p>`).join('');

  setTimeout(() => drawHeatmap('heatmapCanvas', d.grid_results), 200);

  document.getElementById('optResults').scrollIntoView({ behavior: 'smooth' });
}


/* ── Trap Analysis Dashboard ── */
function buildTrapAnalysis(d) {
  const el = document.getElementById('trapAnalysis');
  if (!el) return;

  const trades = d.trades;
  const trapped = trades.filter(t => t.could_have_trapped);
  const notTrapped = trades.filter(t => !t.could_have_trapped);
  const trapRate = trades.length ? (trapped.length / trades.length * 100).toFixed(0) : 0;

  // Timing distribution
  const timeCounts = {};
  const timeImprovements = {};
  trapped.forEach(t => {
    const tm = t.optimal_trap_time || 'Unknown';
    timeCounts[tm] = (timeCounts[tm] || 0) + 1;
    if (!timeImprovements[tm]) timeImprovements[tm] = [];
    timeImprovements[tm].push(t.trap_improvement);
  });

  // Regime breakdown for trapped trades
  const trapRegimes = {};
  trapped.forEach(t => {
    const r = t.regime;
    if (!trapRegimes[r]) trapRegimes[r] = { count: 0, improvements: [], debits: [], combinedPnls: [] };
    trapRegimes[r].count++;
    trapRegimes[r].improvements.push(t.trap_improvement);
    trapRegimes[r].debits.push(t.trap_debit);
    trapRegimes[r].combinedPnls.push(t.trapped_pnl);
  });

  // Trapped vs hold comparison
  const holdPnls = trapped.map(t => t.net_pnl);
  const combinedPnls = trapped.map(t => t.trapped_pnl);
  const avgHold = holdPnls.length ? holdPnls.reduce((a,b) => a+b, 0) / holdPnls.length : 0;
  const avgCombined = combinedPnls.length ? combinedPnls.reduce((a,b) => a+b, 0) / combinedPnls.length : 0;
  const holdWR = holdPnls.length ? (holdPnls.filter(p => p > 0).length / holdPnls.length * 100).toFixed(0) : 0;
  const trapWR = combinedPnls.length ? (combinedPnls.filter(p => p > 0).length / combinedPnls.length * 100).toFixed(0) : 0;

  // Build timing rows
  const timeOrder = ['11:00 AM','12:00 PM','1:00 PM','2:00 PM','2:30 PM','3:00 PM'];
  const maxCount = Math.max(...Object.values(timeCounts), 1);
  const timeRows = timeOrder.filter(t => timeCounts[t]).map(tm => {
    const count = timeCounts[tm] || 0;
    const avgImpr = timeImprovements[tm]
      ? (timeImprovements[tm].reduce((a,b)=>a+b,0) / timeImprovements[tm].length).toFixed(2) : '0';
    const pct = (count / trapped.length * 100).toFixed(0);
    const barW = (count / maxCount * 100).toFixed(0);
    return `<div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">
      <div style="width:70px;font-size:12px;color:var(--text-dim);text-align:right">${tm}</div>
      <div style="flex:1;background:var(--surface);border-radius:4px;height:22px;position:relative;overflow:hidden">
        <div style="width:${barW}%;height:100%;background:linear-gradient(90deg,var(--purple),var(--cyan));border-radius:4px"></div>
        <div style="position:absolute;right:8px;top:2px;font-size:11px;color:var(--text-dim)">${count} (${pct}%)</div>
      </div>
      <div style="width:75px;font-size:12px;font-family:'JetBrains Mono',monospace;color:var(--green);text-align:right">+$${avgImpr}</div>
    </div>`;
  }).join('');

  // Regime trap rows
  const regimeOrder = ['low_vol','normal','elevated','high_vol'];
  const regimeColors = { low_vol: 'var(--green)', normal: 'var(--cyan)', elevated: 'var(--amber)', high_vol: 'var(--red)' };
  const regimeRows = regimeOrder.filter(r => trapRegimes[r]).map(r => {
    const s = trapRegimes[r];
    const avgImpr = (s.improvements.reduce((a,b)=>a+b,0) / s.count).toFixed(2);
    const avgDebit = (s.debits.reduce((a,b)=>a+b,0) / s.count).toFixed(2);
    const avgComb = (s.combinedPnls.reduce((a,b)=>a+b,0) / s.count).toFixed(2);
    const wr = (s.combinedPnls.filter(p => p > 0).length / s.count * 100).toFixed(0);
    return `<tr>
      <td><span style="color:${regimeColors[r]}">${r.replace('_',' ')}</span></td>
      <td>${s.count}</td>
      <td style="color:var(--green)">${wr}%</td>
      <td style="font-family:'JetBrains Mono',monospace">$${avgDebit}</td>
      <td style="font-family:'JetBrains Mono',monospace;color:var(--green)">+$${avgImpr}</td>
      <td style="font-family:'JetBrains Mono',monospace;color:${parseFloat(avgComb) >= 0 ? 'var(--green)' : 'var(--red)'}">$${avgComb}</td>
    </tr>`;
  }).join('');

  el.innerHTML = `
    <div class="grid-4 mt-2">
      <div class="card"><div class="metric purple"><div class="val" style="font-size:24px">${trapRate}%</div><div class="label">Trap Opportunity Rate</div></div>
        <div style="font-size:11px;color:var(--text-muted);text-align:center;margin-top:6px">${trapped.length} of ${trades.length} days SPX crossed profit zone</div>
      </div>
      <div class="card"><div class="metric green"><div class="val" style="font-size:24px">${trapWR}%</div><div class="label">Trapped Win Rate</div></div>
        <div style="font-size:11px;color:var(--text-muted);text-align:center;margin-top:6px">vs ${holdWR}% hold-only on same days</div>
      </div>
      <div class="card"><div class="metric cyan"><div class="val" style="font-size:24px">+$${(avgCombined - avgHold).toFixed(2)}</div><div class="label">Avg Improvement</div></div>
        <div style="font-size:11px;color:var(--text-muted);text-align:center;margin-top:6px">Combined $${avgCombined.toFixed(2)} vs hold $${avgHold.toFixed(2)}</div>
      </div>
      <div class="card"><div class="metric amber"><div class="val" style="font-size:24px">$${trapped.length ? (trapped.reduce((a,t) => a+t.trap_debit, 0) / trapped.length).toFixed(2) : '0'}</div><div class="label">Avg Trap Cost</div></div>
        <div style="font-size:11px;color:var(--text-muted);text-align:center;margin-top:6px">per side, theta-discounted</div>
      </div>
    </div>

    <div class="grid-2 mt-3">
      <div class="card">
        <h3 style="margin-bottom:12px">Optimal Trap Timing</h3>
        <p style="font-size:12px;color:var(--text-muted);margin-bottom:16px">When did the best traps fire? Later = cheaper debit but less certainty SPX stays in zone.</p>
        ${timeRows}
        <div style="margin-top:16px;padding:12px;background:var(--surface2);border-radius:8px">
          <div style="font-size:12px;color:var(--text-dim)">
            <strong style="color:var(--purple)">Insight:</strong> The afternoon (2:00–3:00 PM) is the prime trap window.
            Theta has burned 70-87% of premium, making traps cheap ($0.20–$1.00),
            and SPX has typically settled into its range.
          </div>
        </div>
      </div>

      <div class="card">
        <h3 style="margin-bottom:12px">Trap Performance by Regime</h3>
        <p style="font-size:12px;color:var(--text-muted);margin-bottom:16px">Traps work across all regimes but are cheapest in low vol — and most impactful in elevated vol.</p>
        <table style="font-size:13px">
          <thead><tr><th>Regime</th><th>Traps</th><th>WR</th><th>Avg Cost</th><th>Avg Δ</th><th>Avg Combined</th></tr></thead>
          <tbody>${regimeRows}</tbody>
        </table>
      </div>
    </div>

    <div class="card mt-3" style="border-left:3px solid var(--purple)">
      <h3 style="margin-bottom:12px">Backtested Trap Rules</h3>
      <div class="grid-2" style="gap:16px">
        <div>
          <h4 style="color:var(--green);font-size:13px;margin-bottom:8px">✓ WHEN TO TRAP</h4>
          <div style="font-size:12px;color:var(--text-dim);line-height:2">
            • SPX is between inner wings (put_upper to call_lower)<br>
            • Time is 12:00 PM – 3:00 PM ET (sweet spot: 2:00–2:30)<br>
            • Day range < 1.5× expected move (range compression)<br>
            • VIX regime is low or normal (cheapest trap debits)<br>
            • Use 70% of outer fly width for the trap
          </div>
        </div>
        <div>
          <h4 style="color:var(--red);font-size:13px;margin-bottom:8px">✗ WHEN TO SKIP</h4>
          <div style="font-size:12px;color:var(--text-dim);line-height:2">
            • SPX has already moved outside the profit zone<br>
            • Day range > 1.5× expected move (trending, will leave zone)<br>
            • Before 11:00 AM (trap is expensive, SPX hasn't settled)<br>
            • VIX > 30 (trap debit too high, R:R collapses)<br>
            • Trap debit > 25% of max butterfly profit
          </div>
        </div>
      </div>
    </div>
  `;
}
