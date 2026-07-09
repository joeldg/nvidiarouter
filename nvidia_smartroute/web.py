# @spec[OBSERVABILITY.md#Requirements]
"""
Self-contained web dashboard + prompt playground.

Served by the gateway at ``/dashboard``. Polls ``/metrics`` for live state and
calls ``/explain`` for the playground, which shows not just the answer but why
a request routed the way it did (task scores, confidence, selected model,
cache/fallback, latency, tokens, cost). No external assets — everything inline.
"""

# @spec[OBSERVABILITY.md#Requirements]
DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>NVIDIA SmartRoute — Dashboard</title>
<style>
  :root{
    --bg:#0b0f0c; --panel:#12181300; --card:#141b16; --card2:#0f150f;
    --line:#243026; --text:#e8f0e8; --muted:#8aa08c; --accent:#76b900;
    --accent2:#a6e22e; --warn:#e6b800; --bad:#e05a5a; --good:#76b900;
    --radius:14px; --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
  }
  *{box-sizing:border-box}
  body{margin:0;background:radial-gradient(1200px 600px at 70% -10%,#12261066,transparent),
       var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
  a{color:var(--accent2)}
  header{display:flex;align-items:center;gap:14px;padding:16px 22px;border-bottom:1px solid var(--line)}
  header .logo{width:12px;height:12px;border-radius:50%;background:var(--accent);box-shadow:0 0 14px var(--accent)}
  header h1{font-size:16px;margin:0;letter-spacing:.3px}
  header .sub{color:var(--muted);font-size:12px}
  header .status{margin-left:auto;display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted)}
  .dot{width:9px;height:9px;border-radius:50%;background:var(--bad);transition:.3s}
  .dot.up{background:var(--good);box-shadow:0 0 10px var(--good)}
  main{max-width:1200px;margin:0 auto;padding:22px;display:flex;flex-direction:column;gap:18px}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
  .card{background:linear-gradient(180deg,var(--card),var(--card2));border:1px solid var(--line);
        border-radius:var(--radius);padding:14px 16px}
  .card .k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}
  .card .v{font-size:26px;font-weight:650;margin-top:4px;font-variant-numeric:tabular-nums}
  .card .v small{font-size:13px;color:var(--muted);font-weight:400}
  .panel{background:linear-gradient(180deg,var(--card),var(--card2));border:1px solid var(--line);
         border-radius:var(--radius);padding:16px 18px}
  .panel h2{font-size:12px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin:0 0 12px}
  canvas{width:100%;height:120px;display:block}
  table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
  th,td{text-align:right;padding:7px 10px;border-bottom:1px solid var(--line);font-size:13px}
  th:first-child,td:first-child{text-align:left}
  th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.06em}
  td.model{font-family:var(--mono);font-size:12px}
  .pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;border:1px solid var(--line)}
  .pill.open{color:var(--bad);border-color:#5a2222;background:#2a1414}
  .pill.closed{color:var(--good);border-color:#274016;background:#16220f}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  @media(max-width:900px){.grid2{grid-template-columns:1fr}}
  #log{font-family:var(--mono);font-size:12px;height:280px;overflow:auto;display:flex;flex-direction:column-reverse}
  #log div{padding:3px 0;border-bottom:1px dashed #1c261d}
  #log .task{color:var(--accent2)} #log .mdl{color:var(--accent)} #log .t{color:var(--muted)}
  textarea{width:100%;min-height:78px;background:var(--card2);color:var(--text);border:1px solid var(--line);
           border-radius:10px;padding:10px 12px;font:13px/1.5 var(--mono);resize:vertical}
  .row{display:flex;gap:10px;align-items:center;margin-top:10px}
  button{background:var(--accent);color:#06210a;border:0;border-radius:10px;padding:9px 16px;
         font-weight:700;cursor:pointer}
  button:disabled{opacity:.5;cursor:default}
  .answer{white-space:pre-wrap;background:var(--card2);border:1px solid var(--line);border-radius:10px;
          padding:12px;margin-top:12px;font-family:var(--mono);font-size:13px}
  .explain{margin-top:12px;display:flex;flex-direction:column;gap:8px}
  .badge{display:inline-flex;gap:8px;align-items:center;background:#16220f;border:1px solid #274016;
         border-radius:999px;padding:4px 12px;font-family:var(--mono);font-size:12px;color:var(--accent2)}
  .bars{display:flex;flex-direction:column;gap:5px;margin-top:6px}
  .bar{display:grid;grid-template-columns:130px 1fr 34px;gap:8px;align-items:center;font-size:12px}
  .bar .track{height:8px;background:#1c261d;border-radius:6px;overflow:hidden}
  .bar .fill{height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2))}
  .meta{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:12px;font-family:var(--mono)}
  .flags{display:flex;gap:6px;flex-wrap:wrap}
  .flag{font-size:11px;padding:1px 8px;border-radius:999px;border:1px solid var(--line)}
  .flag.on{color:var(--accent2);border-color:#274016;background:#16220f}
</style>
</head>
<body>
<header>
  <div class="logo"></div>
  <div><h1>NVIDIA SmartRoute</h1><div class="sub">OpenAI-compatible NIM gateway</div></div>
  <div class="status"><span id="dot" class="dot"></span><span id="statusText">connecting…</span></div>
</header>
<main>
  <div class="cards" id="cards"></div>

  <div class="panel">
    <h2>Requests / sec</h2>
    <canvas id="chart"></canvas>
  </div>
  <div class="panel"><h2>PARKOUR recent runs</h2><div id="parkour"></div></div>

  <div class="panel">
    <h2>Model performance</h2>
    <table><thead><tr>
      <th>Model</th><th>Params</th><th>Reqs</th><th>Avg ms</th><th>Tok/s</th>
      <th>Max t/s</th><th>Cost $</th><th>Errors</th><th>Circuit</th>
    </tr></thead><tbody id="models"></tbody></table>
  </div>

  <div class="grid2">
    <div class="panel">
      <h2>Routing log</h2>
      <div id="log"></div>
    </div>
    <div class="panel">
      <h2>Playground — route &amp; explain</h2>
      <textarea id="prompt" placeholder="Ask something… e.g. 'Write a Python function to sort a list' or 'What is 17 * 23?'"></textarea>
      <div class="row">
        <button id="send" onclick="explain()">Route &amp; Explain</button>
        <span id="pnote" class="sub" style="color:var(--muted)"></span>
      </div>
      <div id="result"></div>
    </div>
  </div>
</main>

<script>
const HISTORY = 90;
let hist = new Array(HISTORY).fill(0), lastTotal = null, lastT = null;

function fmt(n, d=0){ return (n==null?0:n).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d}); }

function drawChart(){
  const c = document.getElementById('chart'), dpr = window.devicePixelRatio||1;
  const w = c.clientWidth, h = c.clientHeight; c.width=w*dpr; c.height=h*dpr;
  const x = c.getContext('2d'); x.scale(dpr,dpr); x.clearRect(0,0,w,h);
  const max = Math.max(1, ...hist); const step = w/(HISTORY-1);
  // grid
  x.strokeStyle='#1c261d'; x.lineWidth=1;
  for(let i=0;i<=3;i++){ const gy=h*i/3; x.beginPath();x.moveTo(0,gy);x.lineTo(w,gy);x.stroke(); }
  // area + line
  x.beginPath(); x.moveTo(0,h);
  hist.forEach((v,i)=>{ const px=i*step, py=h-(v/max)*(h-6)-3; x.lineTo(px,py); });
  x.lineTo(w,h); x.closePath();
  const g=x.createLinearGradient(0,0,0,h); g.addColorStop(0,'#76b90055'); g.addColorStop(1,'#76b90000');
  x.fillStyle=g; x.fill();
  x.beginPath();
  hist.forEach((v,i)=>{ const px=i*step, py=h-(v/max)*(h-6)-3; i?x.lineTo(px,py):x.moveTo(px,py); });
  x.strokeStyle='#a6e22e'; x.lineWidth=2; x.stroke();
  x.fillStyle='#8aa08c'; x.font='11px ui-monospace,monospace'; x.fillText('peak '+max.toFixed(1)+'/s',8,14);
}

function card(k,v,sub){ return `<div class="card"><div class="k">${k}</div><div class="v">${v}${sub?` <small>${sub}</small>`:''}</div></div>`; }

async function poll(){
  let d;
  try{ d = await (await fetch('/metrics')).json(); }
  catch(e){ document.getElementById('dot').className='dot';
            document.getElementById('statusText').textContent='gateway unreachable'; return; }
  document.getElementById('dot').className='dot up';
  document.getElementById('statusText').textContent='live · '+Math.floor(d.uptime_seconds)+'s uptime';

  // rate
  const now=Date.now()/1000, total=d.total_requests||0;
  if(lastTotal!=null){ const dt=Math.max(now-lastT,1e-6); hist.push(Math.max(0,(total-lastTotal)/dt)); hist=hist.slice(-HISTORY); }
  lastTotal=total; lastT=now; drawChart();

  const cache=d.cache||{}, conc=d.concurrency||{}, bud=d.budget||{};
  document.getElementById('cards').innerHTML =
    card('Requests', fmt(total)) +
    card('Active conns', fmt(d.active_connections)) +
    card('In-flight', fmt(conc.inflight)+' <small>/'+fmt(conc.max_inflight)+'</small>') +
    card('Cache hit', fmt((cache.hit_rate||0)*100)+'%') +
    card('Spend', '$'+fmt(d.total_cost_usd||0,4)) +
    card('Models', fmt((d.models||[]).length));

  document.getElementById('models').innerHTML = (d.models||[])
    .sort((a,b)=>b.request_count-a.request_count)
    .map(m=>{
      const circ=(d.circuits||{})[m.model_id];
      const pill = circ?`<span class="pill open">${circ}</span>`:`<span class="pill closed">ok</span>`;
      const p = m.parameters_b? m.parameters_b.toFixed(0)+'B':'?';
      return `<tr><td class="model">${m.model_id}</td><td>${p}</td><td>${fmt(m.request_count)}</td>
        <td>${fmt(m.avg_latency_ms)}</td><td>${fmt(m.throughput_tps,1)}</td><td>${fmt(m.max_tps,1)}</td>
        <td>${fmt(m.total_cost_usd,4)}</td><td>${fmt(m.error_count)}</td><td>${pill}</td></tr>`;
    }).join('') || '<tr><td colspan="9" style="color:var(--muted)">no traffic yet</td></tr>';
  const pk=d.parkour||{};
  document.getElementById('parkour').innerHTML=(pk.recent_runs||[]).slice(-10).reverse().map(r=>
    `<div><span class="t">${r.run_id}</span> ${r.outcome} · ${r.nodes} nodes · ${r.tokens} tokens</div>`
  ).join('') || '<span style="color:var(--muted)">no PARKOUR runs yet</span>';

  const log=document.getElementById('log');
  log.innerHTML = (d.routing_log||[]).slice(-60).map(e=>{
    const t=new Date(e.timestamp*1000).toLocaleTimeString();
    return `<div><span class="t">${t}</span> <span class="task">${e.task_type}</span> → <span class="mdl">${e.model}</span> <span class="t">(${e.confidence})</span></div>`;
  }).join('');
}

async function explain(){
  const btn=document.getElementById('send'), note=document.getElementById('pnote');
  const prompt=document.getElementById('prompt').value.trim();
  if(!prompt) return;
  btn.disabled=true; note.textContent='routing…';
  const res=document.getElementById('result');
  try{
    const r=await fetch('/explain',{method:'POST',headers:{'content-type':'application/json'},
      body:JSON.stringify({messages:[{role:'user',content:prompt}]})});
    const d=await r.json();
    if(!r.ok){ res.innerHTML=`<div class="answer" style="color:var(--bad)">${(d.error&&d.error.message)||'error'}</div>`; return; }
    const rt=d.routing||{}, scores=rt.scores||{};
    const maxS=Math.max(1,...Object.values(scores));
    const bars=Object.entries(scores).sort((a,b)=>b[1]-a[1]).slice(0,6).map(([k,v])=>
      `<div class="bar"><span>${k}</span><span class="track"><span class="fill" style="width:${(v/maxS)*100}%"></span></span><span>${v.toFixed(1)}</span></div>`).join('');
    const flag=(on,txt)=>`<span class="flag ${on?'on':''}">${txt}</span>`;
    res.innerHTML = `
      <div class="answer">${(d.answer||'(no content)').replace(/</g,'&lt;')}</div>
      <div class="explain">
        <div><span class="badge">▶ ${rt.selected_model} · ${rt.parameters_b?rt.parameters_b.toFixed(0)+'B':'?'}</span></div>
        <div class="meta">
          <span>task: <b style="color:var(--accent2)">${rt.task_type}</b></span>
          <span>confidence: ${(rt.confidence*100).toFixed(0)}%</span>
          <span>latency: ${fmt(d.latency_ms)}ms</span>
          <span>tokens: ${fmt((d.usage&&d.usage.total_tokens)||0)}</span>
          <span>cost: $${fmt(d.cost_usd||0,6)}</span>
        </div>
        <div class="flags">
          ${flag(rt.fell_back,'fell back')}
          ${flag(d.cache==='HIT','cache hit')}
          ${flag(rt.autoscaled,'autoscaled')}
        </div>
        <div class="bars">${bars||'<span class="sub">no task signals (default chat)</span>'}</div>
      </div>`;
  }catch(e){ res.innerHTML=`<div class="answer" style="color:var(--bad)">${e}</div>`; }
  finally{ btn.disabled=false; note.textContent=''; }
}

poll(); setInterval(poll, 1500);
window.addEventListener('resize', drawChart);
document.getElementById('prompt').addEventListener('keydown',e=>{ if(e.key==='Enter'&&(e.metaKey||e.ctrlKey)) explain(); });
</script>
</body>
</html>
"""
