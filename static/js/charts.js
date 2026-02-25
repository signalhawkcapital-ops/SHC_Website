/* ── charts.js — Shared canvas drawing utilities ── */

const Colors = {
  bg: '#0a0b0f', surface: '#12131a', surface2: '#1a1b24', surface3: '#22232e',
  border: '#2a2b3a', text: '#e8e9f0', textDim: '#8a8b9a', textMuted: '#5a5b6a',
  accent: '#4f6ef7', green: '#22c55e', red: '#ef4444', amber: '#f59e0b',
  cyan: '#06b6d4', purple: '#a855f7',
};

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  // Fallback: if rect.width is 0 (parent hidden/not laid out), use attribute or parent width
  let w = rect.width;
  if (w < 10) {
    w = parseInt(canvas.getAttribute('width')) || canvas.parentElement?.offsetWidth || 600;
  }
  const h = parseInt(canvas.getAttribute('height')) || 300;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  canvas.style.height = h + 'px';
  return { ctx, w, h };
}

function drawGrid(ctx, pad, w, h, rows, cols, color) {
  color = color || Colors.surface2;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b;
  for (let i = 0; i <= rows; i++) {
    const y = pad.t + (ch / rows) * i;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
  }
  if (cols) {
    for (let i = 0; i <= cols; i++) {
      const x = pad.l + (cw / cols) * i;
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + ch); ctx.stroke();
    }
  }
}

function drawThetaCurve(canvasId, thetaData) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const { ctx, w, h } = setupCanvas(canvas);
  const pad = { t: 20, r: 16, b: 32, l: 44 };
  const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b;

  ctx.clearRect(0, 0, w, h);
  if (!thetaData || !thetaData.length) return;

  // Y-axis: 0% at bottom (nothing decayed) to 100% at top (fully decayed)
  // Grid lines
  ctx.strokeStyle = Colors.surface2;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = pad.t + (ch / 5) * i;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
  }

  // Y-axis labels: "% Premium Decayed"
  ctx.fillStyle = Colors.textDim;
  ctx.font = '10px JetBrains Mono';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 5; i++) {
    const pct = 100 - i * 20;
    const y = pad.t + (ch / 5) * i;
    ctx.fillText(pct + '%', pad.l - 6, y + 4);
  }

  // X-axis time labels
  const times = [
    {t: 0, label: '9:30'}, {t: 0.23, label: '11:00'},
    {t: 0.385, label: '12:00'}, {t: 0.54, label: '1:00'}, {t: 0.69, label: '2:00'},
    {t: 0.846, label: '3:00'}, {t: 1, label: '4:00'},
  ];
  ctx.font = '10px JetBrains Mono';
  ctx.textAlign = 'center';
  ctx.fillStyle = Colors.textDim;
  times.forEach(({t, label}) => {
    const x = pad.l + t * cw;
    ctx.fillText(label, x, h - 6);
    ctx.strokeStyle = Colors.surface2 + '60';
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + ch); ctx.stroke();
  });

  // Trap zone shading: 50%-100% decayed region (top half of chart)
  const y50 = pad.t + ch * 0.5;
  ctx.fillStyle = Colors.green + '08';
  ctx.fillRect(pad.l, pad.t, cw, y50 - pad.t);
  // Label
  ctx.fillStyle = Colors.green + '50';
  ctx.font = '9px Sora';
  ctx.textAlign = 'right';
  ctx.fillText('TRAP ZONE (>50% decayed)', w - pad.r - 4, pad.t + 12);

  // 50% line
  ctx.strokeStyle = Colors.green + '30';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(pad.l, y50); ctx.lineTo(w - pad.r, y50); ctx.stroke();
  ctx.setLineDash([]);

  // 80% line
  const y80 = pad.t + ch * 0.2;
  ctx.strokeStyle = Colors.amber + '30';
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(pad.l, y80); ctx.lineTo(w - pad.r, y80); ctx.stroke();
  ctx.setLineDash([]);

  // Draw gradient fill under curve first
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t + ch);
  thetaData.forEach((d) => {
    const x = pad.l + d.elapsed * cw;
    const decay = Math.min(d.cumulative_decay, 1);
    const y = pad.t + ch - decay * ch;
    ctx.lineTo(x, y);
  });
  ctx.lineTo(pad.l + thetaData[thetaData.length-1].elapsed * cw, pad.t + ch);
  ctx.closePath();
  const fill = ctx.createLinearGradient(0, pad.t, 0, pad.t + ch);
  fill.addColorStop(0, Colors.accent + '18');
  fill.addColorStop(1, Colors.accent + '02');
  ctx.fillStyle = fill;
  ctx.fill();

  // Main decay curve
  ctx.beginPath();
  ctx.lineWidth = 2.5;
  const gradient = ctx.createLinearGradient(pad.l, 0, w - pad.r, 0);
  gradient.addColorStop(0, Colors.cyan);
  gradient.addColorStop(0.5, Colors.green);
  gradient.addColorStop(0.8, Colors.amber);
  gradient.addColorStop(1, Colors.red);
  ctx.strokeStyle = gradient;
  thetaData.forEach((d, i) => {
    const x = pad.l + d.elapsed * cw;
    const y = pad.t + ch - Math.min(d.cumulative_decay, 1) * ch;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Key point annotations on the curve
  const milestones = [
    { pct: 0.50, label: '50%', sublabel: 'traps viable', color: Colors.green },
    { pct: 0.80, label: '80%', sublabel: 'acceleration', color: Colors.amber },
  ];
  milestones.forEach(m => {
    const pt = thetaData.find(d => d.cumulative_decay >= m.pct);
    if (!pt) return;
    const x = pad.l + pt.elapsed * cw;
    const y = pad.t + ch - m.pct * ch;
    // Dot
    ctx.fillStyle = m.color;
    ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.beginPath(); ctx.arc(x, y, 1.5, 0, Math.PI*2); ctx.fill();
    // Label
    ctx.fillStyle = m.color;
    ctx.font = 'bold 9px JetBrains Mono';
    ctx.textAlign = 'left';
    ctx.fillText(m.label, x + 6, y - 1);
    ctx.font = '8px Sora';
    ctx.fillText(m.sublabel, x + 6, y + 9);
  });

  // "Now" marker during market hours
  try {
    const nowET = new Date(new Date().toLocaleString('en-US', { timeZone: 'America/New_York' }));
    const etH = nowET.getHours() + nowET.getMinutes() / 60;
    const mktFrac = (etH - 9.5) / 6.5;
    if (mktFrac > 0 && mktFrac < 1 && nowET.getDay() >= 1 && nowET.getDay() <= 5) {
      const nx = pad.l + mktFrac * cw;
      ctx.strokeStyle = Colors.accent;
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(nx, pad.t); ctx.lineTo(nx, pad.t + ch); ctx.stroke();
      ctx.setLineDash([]);
      // NOW label
      ctx.fillStyle = Colors.accent;
      ctx.font = 'bold 10px Sora';
      ctx.textAlign = 'center';
      ctx.fillText('NOW', nx, pad.t - 4);
      // Dot on curve
      const decay = 1 - Math.pow(1 - mktFrac, 0.58);
      const dotY = pad.t + ch - decay * ch;
      ctx.beginPath(); ctx.arc(nx, dotY, 5, 0, Math.PI*2);
      ctx.fillStyle = Colors.accent; ctx.fill();
      ctx.beginPath(); ctx.arc(nx, dotY, 2, 0, Math.PI*2);
      ctx.fillStyle = '#fff'; ctx.fill();
      ctx.fillStyle = Colors.accent;
      ctx.font = 'bold 10px JetBrains Mono';
      ctx.textAlign = 'left';
      ctx.fillText(`${(decay*100).toFixed(0)}%`, nx + 8, dotY + 4);
    }
  } catch(e) {}
}

function drawPnlMap(canvasId, pnlData, spxOpen, range1, callDebit, putDebit, strat, trapInfo) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const { ctx, w, h } = setupCanvas(canvas);
  const pad = { t: 50, r: 40, b: 52, l: 80 };
  const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b;

  ctx.clearRect(0, 0, w, h);

  if (!pnlData || !pnlData.length) return;

  callDebit = callDebit || 0;
  putDebit = putDebit || 0;
  const totalDebit = callDebit + putDebit;

  // Butterfly P&L at expiry: long lower + long upper, short 2× center
  function flyPnl(lower, center, upper, settlement) {
    return Math.max(0, settlement - lower)
         - 2 * Math.max(0, settlement - center)
         + Math.max(0, settlement - upper);
  }

  // Compute net P&L (butterfly value minus debit paid, ×100 for contract)
  const hasTrap = trapInfo && trapInfo.center && trapInfo.width && trapInfo.debit > 0;
  const trapDebit = hasTrap ? trapInfo.debit : 0;
  const allDebit = totalDebit + trapDebit;

  const netData = pnlData.map(d => {
    const callNet = (d.call_pnl - callDebit) * 100;
    const putNet = (d.put_pnl - putDebit) * 100;
    const batmanNet = (d.total_pnl - totalDebit) * 100;

    let trapNet = 0;
    if (hasTrap) {
      const tl = trapInfo.center - trapInfo.width;
      const tc = trapInfo.center;
      const tu = trapInfo.center + trapInfo.width;
      trapNet = (flyPnl(tl, tc, tu, d.price) - trapDebit) * 100;
    }

    return {
      price: d.price,
      callNet,
      putNet,
      batmanNet,
      trapNet,
      totalNet: batmanNet + (hasTrap ? trapNet + trapDebit * 100 : 0),
      // Combined = batman P&L + trap butterfly value (not double-counting debit)
      combinedNet: batmanNet + (hasTrap ? (trapNet + trapDebit * 100) : 0),
    };
  });

  // Recalculate combinedNet properly: batman debits already subtracted, add trap fly value - trap debit
  netData.forEach(d => {
    if (hasTrap) {
      const tl = trapInfo.center - trapInfo.width;
      const tc = trapInfo.center;
      const tu = trapInfo.center + trapInfo.width;
      const trapValue = flyPnl(tl, tc, tu, d.price) * 100;
      d.combinedNet = d.batmanNet + trapValue - trapDebit * 100;
    } else {
      d.combinedNet = d.batmanNet;
    }
  });

  const prices = netData.map(d => d.price);
  const displayVals = netData.map(d => d.combinedNet);
  const maxProfit = Math.max(...displayVals);
  const maxLoss = Math.min(...displayVals);
  const yBot = Math.min(-500, maxLoss * 1.1, -(allDebit * 100) * 1.2);
  const yTop = maxProfit * 1.12;
  const pMin = yBot, pMax = yTop;
  const pRange = pMax - pMin || 1;
  const priceMin = prices[0], priceMax = prices[prices.length - 1];
  const priceRange = priceMax - priceMin;

  const xOf = (price) => pad.l + ((price - priceMin) / priceRange) * cw;
  const yOf = (val) => pad.t + ch - ((val - pMin) / pRange) * ch;

  // Background
  ctx.fillStyle = Colors.surface + 'c0';
  ctx.fillRect(pad.l, pad.t, cw, ch);

  // Grid
  drawGrid(ctx, pad, w, h, 6, 8, Colors.surface2);

  // 1σ range band
  if (range1) {
    const x1 = Math.max(xOf(range1[0]), pad.l), x2 = Math.min(xOf(range1[1]), w - pad.r);
    ctx.fillStyle = Colors.accent + '0c';
    ctx.fillRect(x1, pad.t, x2 - x1, ch);
    ctx.strokeStyle = Colors.accent + '30';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(x1, pad.t); ctx.lineTo(x1, pad.t + ch); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x2, pad.t); ctx.lineTo(x2, pad.t + ch); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = Colors.accent + '80';
    ctx.font = '10px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText('← 1σ Expected Move →', (x1 + x2) / 2, pad.t + 14);
  }

  // Zero line (breakeven) — prominent
  const zeroY = yOf(0);
  if (zeroY >= pad.t && zeroY <= pad.t + ch) {
    ctx.strokeStyle = '#ffffff40';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(pad.l, zeroY); ctx.lineTo(w - pad.r, zeroY); ctx.stroke();
    // "$0 BREAKEVEN" label
    ctx.fillStyle = '#ffffff70';
    ctx.font = 'bold 9px Sora';
    ctx.textAlign = 'left';
    ctx.fillText('$0  BREAKEVEN', pad.l + 4, zeroY - 5);
  }

  // Profit zone fill (smooth area)
  ctx.beginPath();
  ctx.moveTo(xOf(netData[0].price), zeroY);
  netData.forEach(d => {
    const x = xOf(d.price), y = yOf(Math.max(0, d.combinedNet));
    ctx.lineTo(x, d.combinedNet > 0 ? y : zeroY);
  });
  ctx.lineTo(xOf(netData[netData.length - 1].price), zeroY);
  ctx.closePath();
  const profGrad = ctx.createLinearGradient(0, yOf(maxProfit), 0, zeroY);
  profGrad.addColorStop(0, Colors.green + '35');
  profGrad.addColorStop(1, Colors.green + '05');
  ctx.fillStyle = profGrad;
  ctx.fill();

  // Loss zone fill
  if (allDebit > 0) {
    ctx.beginPath();
    ctx.moveTo(xOf(netData[0].price), zeroY);
    netData.forEach(d => {
      const x = xOf(d.price), y = yOf(Math.min(0, d.combinedNet));
      ctx.lineTo(x, d.combinedNet < 0 ? y : zeroY);
    });
    ctx.lineTo(xOf(netData[netData.length - 1].price), zeroY);
    ctx.closePath();
    ctx.fillStyle = Colors.red + '10';
    ctx.fill();
  }

  // SPX Open marker
  if (spxOpen && spxOpen >= priceMin && spxOpen <= priceMax) {
    const ox = xOf(spxOpen);
    ctx.strokeStyle = Colors.accent + '50';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(ox, pad.t); ctx.lineTo(ox, pad.t + ch); ctx.stroke();
    ctx.setLineDash([]);
    // Label
    ctx.fillStyle = Colors.accent;
    ctx.font = 'bold 11px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText('▼ Open ' + spxOpen.toFixed(0), ox, pad.t - 6);
  }

  // Individual fly lines (subtle)
  [{data: 'callNet', color: '#a78bfa', label: 'Call'}, {data: 'putNet', color: '#f472b6', label: 'Put'}].forEach(fly => {
    ctx.beginPath();
    ctx.strokeStyle = fly.color + '60';
    ctx.lineWidth = 1;
    netData.forEach((d, i) => {
      const x = xOf(d.price), y = yOf(Math.max(pMin, Math.min(pMax, d[fly.data])));
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  // Trap butterfly line (purple, if active)
  if (hasTrap) {
    ctx.beginPath();
    ctx.strokeStyle = Colors.purple + '80';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    netData.forEach((d, i) => {
      const x = xOf(d.price), y = yOf(Math.max(pMin, Math.min(pMax, d.trapNet)));
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // Batman-only line (dimmed when trap is active)
    ctx.beginPath();
    ctx.strokeStyle = Colors.textMuted + '40';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 4]);
    netData.forEach((d, i) => {
      const x = xOf(d.price), y = yOf(Math.max(pMin, Math.min(pMax, d.batmanNet)));
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Combined net P&L — bold green/red line
  ctx.lineWidth = 2.5;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (let i = 1; i < netData.length; i++) {
    const d0 = netData[i-1], d1 = netData[i];
    ctx.beginPath();
    ctx.strokeStyle = d1.combinedNet >= 0 ? Colors.green : Colors.red;
    ctx.moveTo(xOf(d0.price), yOf(Math.max(pMin, Math.min(pMax, d0.combinedNet))));
    ctx.lineTo(xOf(d1.price), yOf(Math.max(pMin, Math.min(pMax, d1.combinedNet))));
    ctx.stroke();
  }

  // Max profit annotations (both peaks)
  const callPeak = netData.filter(d => strat && d.price > spxOpen).reduce((a, b) => a.combinedNet > b.combinedNet ? a : b, {combinedNet: -Infinity});
  const putPeak = netData.filter(d => strat && d.price < spxOpen).reduce((a, b) => a.combinedNet > b.combinedNet ? a : b, {combinedNet: -Infinity});
  [callPeak, putPeak].forEach(peak => {
    if (!peak || !peak.price || peak.combinedNet <= 0) return;
    const px = xOf(peak.price), py = yOf(peak.combinedNet);
    // Dot
    ctx.fillStyle = Colors.green;
    ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.beginPath(); ctx.arc(px, py, 2, 0, Math.PI * 2); ctx.fill();
    // Label with background
    const label = `+$${peak.combinedNet.toFixed(0)}`;
    ctx.font = 'bold 12px JetBrains Mono';
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = Colors.surface + 'e0';
    ctx.fillRect(px - tw/2 - 4, py - 24, tw + 8, 18);
    ctx.strokeStyle = Colors.green + '60';
    ctx.lineWidth = 1;
    ctx.strokeRect(px - tw/2 - 4, py - 24, tw + 8, 18);
    ctx.fillStyle = Colors.green;
    ctx.textAlign = 'center';
    ctx.fillText(label, px, py - 10);
  });

  // Max loss label
  if (allDebit > 0) {
    const maxLossAmt = -allDebit * 100;
    const lossY = yOf(maxLossAmt);
    if (lossY >= pad.t && lossY <= pad.t + ch) {
      ctx.strokeStyle = Colors.red + '40';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 5]);
      ctx.beginPath(); ctx.moveTo(pad.l, lossY); ctx.lineTo(w - pad.r, lossY); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = Colors.red;
      ctx.font = '10px JetBrains Mono';
      ctx.textAlign = 'left';
      ctx.fillText(`Max loss: -$${Math.abs(maxLossAmt).toFixed(0)}`, pad.l + 6, lossY + 14);
    }
  }

  // X-axis price labels (evenly spaced)
  ctx.fillStyle = Colors.textMuted;
  ctx.font = '10px JetBrains Mono';
  ctx.textAlign = 'center';
  const xSteps = 8;
  for (let i = 0; i <= xSteps; i++) {
    const p = priceMin + (priceRange / xSteps) * i;
    const x = pad.l + (cw / xSteps) * i;
    ctx.fillText(Math.round(p), x, pad.t + ch + 14);
    // Tick mark
    ctx.strokeStyle = Colors.surface3;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, pad.t + ch); ctx.lineTo(x, pad.t + ch + 4); ctx.stroke();
  }
  // "SPX Settlement Price" axis label
  ctx.fillStyle = Colors.textDim;
  ctx.font = '10px Sora';
  ctx.fillText('SPX Settlement Price', pad.l + cw / 2, pad.t + ch + 28);

  // Strike markers (colored tick marks above X-axis labels)
  if (strat) {
    const strikes = [
      { price: strat.put_lower, label: 'PL', color: '#f472b6' },
      { price: strat.put_center, label: 'PC', color: '#f472b6' },
      { price: strat.put_upper, label: 'PU', color: '#f472b6' },
      { price: strat.call_lower, label: 'CL', color: '#a78bfa' },
      { price: strat.call_center, label: 'CC', color: '#a78bfa' },
      { price: strat.call_upper, label: 'CU', color: '#a78bfa' },
    ];
    // Add trap strikes if active
    if (hasTrap) {
      const tl = trapInfo.center - trapInfo.width;
      const tu = trapInfo.center + trapInfo.width;
      strikes.push(
        { price: tl, label: 'TL', color: Colors.purple },
        { price: trapInfo.center, label: 'TC', color: Colors.purple },
        { price: tu, label: 'TU', color: Colors.purple },
      );
    }
    strikes.forEach(s => {
      if (s.price >= priceMin && s.price <= priceMax) {
        const x = xOf(s.price);
        // Small colored triangle at chart bottom
        ctx.fillStyle = s.color + '90';
        ctx.beginPath();
        ctx.moveTo(x - 3, pad.t + ch);
        ctx.lineTo(x + 3, pad.t + ch);
        ctx.lineTo(x, pad.t + ch - 6);
        ctx.closePath();
        ctx.fill();
        // Subtle vertical guide
        ctx.setLineDash([2, 8]);
        ctx.strokeStyle = s.color + '15';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + ch - 6); ctx.stroke();
        ctx.setLineDash([]);
        // Label below axis
        ctx.fillStyle = s.color + '70';
        ctx.font = '8px JetBrains Mono';
        ctx.textAlign = 'center';
        ctx.fillText(s.label, x, pad.t + ch + 40);
      }
    });
  }

  // Y-axis labels
  ctx.font = '11px JetBrains Mono';
  ctx.textAlign = 'right';
  const niceStep = getNiceStep(pRange, 6);
  const yStart = Math.ceil(pMin / niceStep) * niceStep;
  const drawnValues = new Set();
  for (let v = yStart; v <= pMax; v += niceStep) {
    const y = yOf(v);
    if (y < pad.t - 5 || y > pad.t + ch + 5) continue;
    const label = v >= 0 ? `+$${v.toFixed(0)}` : `-$${Math.abs(v).toFixed(0)}`;
    ctx.fillStyle = v > 0 ? Colors.green + 'b0' : v < 0 ? Colors.red + 'b0' : '#ffffffb0';
    ctx.fillText(label, pad.l - 8, y + 4);
    drawnValues.add(Math.round(v));
  }
  // Force $0 on Y-axis if not already drawn
  if (!drawnValues.has(0) && zeroY >= pad.t && zeroY <= pad.t + ch) {
    ctx.fillStyle = '#ffffffb0';
    ctx.font = 'bold 11px JetBrains Mono';
    ctx.textAlign = 'right';
    ctx.fillText('$0', pad.l - 8, zeroY + 4);
  }

  // Legend
  ctx.font = '11px Sora';
  ctx.textAlign = 'right';
  const lx = w - pad.r - 8;
  ctx.fillStyle = Colors.green; ctx.fillText('━ Combined Net', lx, pad.t + 16);
  ctx.fillStyle = '#a78bfa80'; ctx.fillText('─ Call Fly', lx, pad.t + 30);
  ctx.fillStyle = '#f472b680'; ctx.fillText('─ Put Fly', lx, pad.t + 44);
  if (hasTrap) {
    ctx.fillStyle = Colors.purple + '90'; ctx.fillText('┄ Trap Fly', lx, pad.t + 58);
  }

  // Debit & R:R info header
  if (allDebit > 0) {
    ctx.fillStyle = Colors.amber;
    ctx.font = 'bold 11px Sora';
    ctx.textAlign = 'left';
    const rr = maxProfit > 0 ? (maxProfit / (allDebit * 100)).toFixed(1) : '—';
    let debitLabel = `Debit: $${(allDebit * 100).toFixed(0)}/ct`;
    if (hasTrap) debitLabel += ` (batman $${(totalDebit*100).toFixed(0)} + trap $${(trapDebit*100).toFixed(0)})`;
    ctx.fillText(`${debitLabel}  ·  R:R ${rr}:1`, pad.l, pad.t - 12);
  }
}

function getNiceStep(range, targetSteps) {
  const raw = range / targetSteps;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  let nice;
  if (norm <= 1.5) nice = 1;
  else if (norm <= 3.5) nice = 2;
  else if (norm <= 7.5) nice = 5;
  else nice = 10;
  return nice * mag;
}

function drawEquityCurve(canvasId, trades) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const { ctx, w, h } = setupCanvas(canvas);
  const pad = { t: 30, r: 20, b: 40, l: 60 };
  const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b;

  ctx.clearRect(0, 0, w, h);
  drawGrid(ctx, pad, w, h, 5);

  if (!trades || !trades.length) return;

  let cum = 0;
  const points = trades.map((t, i) => { cum += t.net_pnl; return { x: i, y: cum }; });
  const maxY = Math.max(...points.map(p => p.y), 0);
  const minY = Math.min(...points.map(p => p.y), 0);
  const range = maxY - minY || 1;

  // Zero line
  const zeroY = pad.t + ch - ((0 - minY) / range) * ch;
  ctx.strokeStyle = Colors.border;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(pad.l, zeroY); ctx.lineTo(w - pad.r, zeroY); ctx.stroke();
  ctx.setLineDash([]);

  // Equity line
  const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + ch);
  grad.addColorStop(0, Colors.green);
  grad.addColorStop(1, Colors.accent);
  ctx.beginPath();
  ctx.strokeStyle = grad;
  ctx.lineWidth = 2;
  points.forEach((p, i) => {
    const x = pad.l + (p.x / (points.length - 1)) * cw;
    const y = pad.t + ch - ((p.y - minY) / range) * ch;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Fill under
  const lastX = pad.l + cw;
  ctx.lineTo(lastX, zeroY);
  ctx.lineTo(pad.l, zeroY);
  ctx.closePath();
  const fill = ctx.createLinearGradient(0, pad.t, 0, pad.t + ch);
  fill.addColorStop(0, Colors.green + '20');
  fill.addColorStop(1, Colors.green + '00');
  ctx.fillStyle = fill;
  ctx.fill();

  // Labels
  ctx.fillStyle = Colors.textMuted;
  ctx.font = '11px JetBrains Mono';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const v = minY + (range / 4) * (4 - i);
    const y = pad.t + (ch / 4) * i;
    ctx.fillText(v.toFixed(1), pad.l - 8, y + 4);
  }
  ctx.textAlign = 'center';
  ctx.fillText('Trade #', w / 2, h - 5);
}

function drawHistogram(canvasId, pnls, bins) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const { ctx, w, h } = setupCanvas(canvas);
  const pad = { t: 20, r: 20, b: 40, l: 50 };
  const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b;

  ctx.clearRect(0, 0, w, h);

  if (!pnls || !pnls.length) return;

  bins = bins || 30;
  const min = Math.min(...pnls);
  const max = Math.max(...pnls);
  const binW = (max - min) / bins;
  const hist = new Array(bins).fill(0);
  pnls.forEach(p => {
    let b = Math.floor((p - min) / binW);
    if (b >= bins) b = bins - 1;
    if (b < 0) b = 0;
    hist[b]++;
  });
  const maxCount = Math.max(...hist);

  const barW = cw / bins;
  hist.forEach((count, i) => {
    const x = pad.l + i * barW;
    const barH = (count / maxCount) * ch;
    const val = min + i * binW;
    ctx.fillStyle = val < 0 ? Colors.red + '80' : Colors.green + '80';
    ctx.fillRect(x + 1, pad.t + ch - barH, barW - 2, barH);
  });

  // Zero line
  const zeroX = pad.l + ((0 - min) / (max - min)) * cw;
  if (zeroX > pad.l && zeroX < w - pad.r) {
    ctx.strokeStyle = Colors.amber;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(zeroX, pad.t); ctx.lineTo(zeroX, pad.t + ch); ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.fillStyle = Colors.textMuted;
  ctx.font = '11px JetBrains Mono';
  ctx.textAlign = 'center';
  ctx.fillText('P&L per Trade', w / 2, h - 5);
}

function drawMonthlyBars(canvasId, monthlyStats) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const { ctx, w, h } = setupCanvas(canvas);
  const pad = { t: 20, r: 20, b: 50, l: 50 };
  const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b;

  ctx.clearRect(0, 0, w, h);

  const months = Object.keys(monthlyStats).sort();
  if (!months.length) return;

  const vals = months.map(m => monthlyStats[m].pnl);
  const maxAbs = Math.max(Math.abs(Math.min(...vals)), Math.abs(Math.max(...vals)), 1);
  const barW = Math.min(40, cw / months.length - 4);
  const zeroY = pad.t + ch / 2;

  // Zero line
  ctx.strokeStyle = Colors.border;
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, zeroY); ctx.lineTo(w - pad.r, zeroY); ctx.stroke();

  months.forEach((m, i) => {
    const pnl = monthlyStats[m].pnl;
    const x = pad.l + (i + 0.5) * (cw / months.length) - barW / 2;
    const barH = (Math.abs(pnl) / maxAbs) * (ch / 2);

    ctx.fillStyle = pnl >= 0 ? Colors.green + 'a0' : Colors.red + 'a0';
    if (pnl >= 0) {
      ctx.fillRect(x, zeroY - barH, barW, barH);
    } else {
      ctx.fillRect(x, zeroY, barW, barH);
    }

    // Label
    ctx.fillStyle = Colors.textMuted;
    ctx.font = '9px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.save();
    ctx.translate(x + barW / 2, h - 8);
    ctx.rotate(-Math.PI / 4);
    ctx.fillText(m, 0, 0);
    ctx.restore();
  });
}

function drawHeatmap(canvasId, gridResults) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const { ctx, w, h } = setupCanvas(canvas);
  const pad = { t: 40, r: 30, b: 60, l: 70 };
  const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b;

  ctx.clearRect(0, 0, w, h);

  if (!gridResults || !gridResults.length) return;

  const widths = [...new Set(gridResults.map(r => r.width_pct))].sort((a, b) => a - b);
  const gaps = [...new Set(gridResults.map(r => r.gap_pct))].sort((a, b) => a - b);
  const sharpes = gridResults.map(r => r.sharpe);
  const maxS = Math.max(...sharpes);
  const minS = Math.min(...sharpes);
  const rangeS = maxS - minS || 1;

  const cellW = cw / widths.length;
  const cellH = ch / gaps.length;

  gridResults.forEach(r => {
    const xi = widths.indexOf(r.width_pct);
    const yi = gaps.indexOf(r.gap_pct);
    const norm = (r.sharpe - minS) / rangeS;

    // Color: red (low) → amber (mid) → green (high)
    let color;
    if (norm < 0.5) {
      const t = norm * 2;
      color = `rgb(${Math.round(239 - 171 * t)},${Math.round(68 + 90 * t)},${Math.round(68 - 57 * t)})`;
    } else {
      const t = (norm - 0.5) * 2;
      color = `rgb(${Math.round(68 - 34 * t)},${Math.round(158 + 39 * t)},${Math.round(11 + 83 * t)})`;
    }

    ctx.fillStyle = color + 'c0';
    ctx.fillRect(pad.l + xi * cellW + 1, pad.t + yi * cellH + 1, cellW - 2, cellH - 2);

    // Value
    ctx.fillStyle = Colors.text;
    ctx.font = '10px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText(r.sharpe.toFixed(1), pad.l + xi * cellW + cellW / 2, pad.t + yi * cellH + cellH / 2 + 4);
  });

  // Axis labels
  ctx.fillStyle = Colors.textDim;
  ctx.font = '11px JetBrains Mono';
  ctx.textAlign = 'center';
  widths.forEach((v, i) => {
    ctx.fillText(v.toFixed(2), pad.l + i * cellW + cellW / 2, h - 15);
  });
  ctx.fillText('Butterfly Width %', w / 2, h - 2);

  ctx.textAlign = 'right';
  gaps.forEach((v, i) => {
    ctx.fillText(v.toFixed(2), pad.l - 8, pad.t + i * cellH + cellH / 2 + 4);
  });
  ctx.save();
  ctx.translate(12, pad.t + ch / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillText('Gap %', 0, 0);
  ctx.restore();

  // Title
  ctx.fillStyle = Colors.textDim;
  ctx.font = '12px Sora';
  ctx.textAlign = 'center';
  ctx.fillText('Sharpe Ratio by Parameter Combination', w / 2, 16);
}
