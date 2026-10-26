(function(){'use strict';
const D=window.BAUMV8_DATA,M=window.BaumModel,$=id=>document.getElementById(id),fmt=x=>Math.round(x).toLocaleString('de-CH'),one=x=>Number(x).toLocaleString('de-CH',{minimumFractionDigits:1,maximumFractionDigits:1});
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const AGE=D.age_labels,AGE10=M.AGE10_LABELS; let YEAR=2030,TARGET=17000,TARGET_YEAR=2030,MODEL=null,GOAL=Array.from({length:18},(_,i)=>i<12?1:0),task=0,selectedTab='entwicklung',cached=null;
const species=Object.keys(D.species_counts).sort((a,b)=>D.species_counts[b]-D.species_counts[a]);
const first=['Quercus robur','Acer campestre','Tilia cordata','Carpinus betulus','Quercus cerris'];
let MIX=first.map((s,i)=>({s:species.includes(s)?s:species[i],weight:[25,20,20,20,15][i]}));
const inputIDs=['replace','delay','annual','endExtra','replMode'];
const CURRENT_AGES_10=(()=>{const a=Array(18).fill(0);D.initial.forEach(([sp,age,n])=>a[M.ageClass10(age)]+=n);return a;})();
function settings(){return {replace:+$('replace').value,delay:+$('delay').value,annual:+$('annual').value,endExtra:+$('endExtra').value,replMode:$('replMode').value,mix:MIX.map(x=>({...x}))};}
function pct(v){return one(v*100)+' %'};
function keepYear(){YEAR=+$('yearSlider').value;$('yearOutput').textContent=YEAR;$('kYearLabel').textContent=YEAR;}
function calc(){keepYear();const S=settings();MODEL=M.run(D,S);cached=S;draw();}
function redraw(){keepYear();if(!MODEL)return;draw();}
function laterCalc(){if(task)cancelAnimationFrame(task);task=requestAnimationFrame(()=>{task=0;calc();});}
function download(text,name,mime){const blob=new Blob([text],{type:mime}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function mainChart(){const series=MODEL.years,ref=D.years,n=series.length,W=1060,H=365,L=74,R=25,T=30,B=42;
let min=Math.min(TARGET,...series.map(x=>x.n),...ref.map(x=>x.p05)),max=Math.max(TARGET,...series.map(x=>x.n),...ref.map(x=>x.p95));min=Math.floor((min-180)/500)*500;max=Math.ceil((max+180)/500)*500;
const x=i=>L+(W-L-R)*i/(n-1),y=v=>T+(H-T-B)*(max-v)/(max-min),pts=(a,key)=>a.map((z,i)=>`${x(i).toFixed(2)},${y(z[key]).toFixed(2)}`).join(' ');
let band=ref.map((v,i)=>`${x(i)},${y(v.p95)}`).concat(ref.slice().reverse().map((v,j)=>`${x(n-1-j)},${y(v.p05)}`)).join(' ');
let grid='',ticks=5;for(let i=0;i<=ticks;i++){let q=min+(max-min)*i/ticks;grid+=`<line x1="${L}" x2="${W-R}" y1="${y(q)}" y2="${y(q)}" stroke="#e8e5de"/><text x="${L-14}" y="${y(q)+4}" text-anchor="end" fill="#8e8981" font-size="12">${fmt(q)}</text>`;}
let labels='';for(let year of [2026,2046,2066,2086,2106,2126]){let idx=year-2026;labels+=`<text x="${x(idx)}" y="${H-12}" text-anchor="middle" font-size="11" fill="#817d76">${year}</text>`;}
const ii=YEAR-2026,xx=x(ii),yy=y(series[ii].n);
$('mainChart').innerHTML=`<svg viewBox="0 0 ${W} ${H}" aria-label="Planung ${fmt(series[ii].n)} Bäume ${YEAR}" role="img" xmlns="http://www.w3.org/2000/svg"><polygon points="${band}" fill="#c5c7c6" opacity=".30"/>${grid}${labels}<polyline points="${pts(ref,'mean')}" stroke="#929593" stroke-width="3.5" fill="none" stroke-linejoin="round"/><line x1="${L}" x2="${W-R}" y1="${y(TARGET)}" y2="${y(TARGET)}" stroke="#4f8b6f" stroke-width="2" stroke-dasharray="7 7"/><text x="${W-R-8}" y="${y(TARGET)-9}" fill="#4f8b6f" text-anchor="end" font-size="12">Ziel ${fmt(TARGET)}</text><polyline points="${pts(series,'n')}" stroke="#e1000f" stroke-width="4" fill="none" stroke-linejoin="round"/><line x1="${xx}" y1="${T}" x2="${xx}" y2="${H-B}" stroke="#9d938b" stroke-dasharray="3 6"/><circle cx="${xx}" cy="${yy}" r="7" stroke="#fff" stroke-width="3" fill="#e1000f"/><text x="${xx}" y="${T-11}" font-size="13" text-anchor="middle" font-weight="700" fill="#a92d35">${YEAR}</text></svg>`;
const diff=series[ii].n-ref[ii].mean;$('chartComparison').textContent=`Szenario ggü. Python-Referenz ${diff<0?'−':'+'}${fmt(Math.abs(diff))} Bäume (${YEAR})`;
}
function bars(id,rows,color){const max=Math.max(1,...rows.map(r=>r.n));$(id).innerHTML=rows.map(r=>`<div class="barrow ${color||''}"><span class="barlabel" title="${esc(r.name)}">${esc(r.name)}</span><span class="barbg"><i class="barfill" style="display:block;width:${Math.max(0,100*r.n/max).toFixed(2)}%"></i></span><span class="barvalue">${fmt(r.n)}</span></div>`).join('');}
function getDeficitAnalysis(){
  const method=$('normalMethod').value,h=+$('horizon').value;
  const dest=M.normalwaldReference10(D,method,h,GOAL);
  const actual=MODEL.ageYears10[YEAR-2026];
  const total=$('deficitBasis').value==='target'?TARGET:actual.reduce((a,b)=>a+b,0);
  return M.normalwaldDeficits(actual,dest,total);
}
function renderDeficits(){
  const x=getDeficitAnalysis(),same=$('deficitBasis').value==='same';
  $('deficitMissing').textContent=fmt(x.missing);
  $('deficitExcess').textContent=fmt(x.excess);
  $('deficitLargest').textContent=fmt(x.largestMissing.deficit);
  $('deficitLargestDetail').textContent=x.largestMissing.deficit>0?AGE10[x.largestMissing.index]+' Jahre':'Keine Klasse mit Defizit';
  $('deficitContext').textContent=`Vergleich für ${YEAR}: Szenario ${fmt(x.currentTotal)} Bäume · Sollverteilung auf ${fmt(x.referenceTotal)} Bäume bezogen${same?' (reiner Strukturvergleich)':' (deine Zielvorgabe)'} · Altersgruppen aus aggregierten 5-Jahres-Kohorten.`;
  const max=Math.max(1,...x.rows.map(r=>Math.abs(r.difference)));
  $('deficitRows').innerHTML=x.rows.map((r,i)=>{
    const isDef=r.difference>0.5,isExtra=r.difference < -0.5;
    const category=isDef?'deficit':isExtra?'excess':'even';
    const label=isDef?'− '+fmt(r.deficit)+' fehlen':isExtra?'+ '+fmt(r.surplus)+' Überhang':'≈ ausgeglichen';
    const w=(Math.abs(r.difference)/max*50).toFixed(2);
    const chart=isDef?`<i class="right" style="width:${w}%"></i>`:isExtra?`<i class="left" style="width:${w}%"></i>`:'';
    return `<div class="deficitRow ${category}"><div class="deficitAge">${esc(AGE10[i])} J.</div>`+
      `<div class="deficitBar" role="img" aria-label="${esc(AGE10[i])} Jahre: ${esc(label)}"><span class="zero"></span>${chart}</div>`+
      `<div class="deficitNum">${fmt(r.current)}</div><div class="deficitNum">${fmt(r.target)}</div>`+
      `<div class="deficitDelta">${esc(label)}</div></div>`;
  }).join('');
  const gap=x.referenceTotal-x.currentTotal;
  $('deficitBalance').textContent=(Math.abs(gap)<0.5?'Die Gesamtsumme ist gleich – Defizite und Überhänge gleichen sich rechnerisch aus.':
    `Dein Zielbestand liegt ${gap>0?'um '+fmt(gap)+' Bäume höher':'um '+fmt(Math.abs(gap))+' Bäume tiefer'} als der modellierte Bestand (${YEAR}).`);
}
function ageNormalPlot(target){
  const W=1110,H=353,L=60,R=18,T=22,B=82,counts=MODEL.ageYears10[YEAR-2026],old=CURRENT_AGES_10;
  const total=counts.reduce((a,b)=>a+b,0),base=old.reduce((a,b)=>a+b,0);
  const cur=counts.map(x=>total?x/total:0),was=old.map(x=>base?x/base:0);
  const top=Math.max(.10,Math.ceil(Math.max(...cur,...was,...target)*100/5)*5/100);
  const py=p=>T+(H-T-B)*(1-p/top),px=i=>L+(W-L-R)*(i+.5)/18,step=(W-L-R)/18;
  let grid='';for(let y=0;y<=top+.00001;y+=.05){const yy=py(y);grid+=`<line x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}" stroke="#e5e2dc" stroke-dasharray="4 6"/><text x="${L-10}" y="${yy+4}" font-size="11" text-anchor="end" fill="#817c74">${Math.round(100*y)} %</text>`;}
  let bars='',labels='';for(let i=0;i<18;i++){
    const x=px(i),bw=Math.min(18,step*.3),greyY=py(was[i]),redY=py(cur[i]),floor=py(0);
    bars+=`<rect x="${(x-bw-1).toFixed(1)}" y="${greyY}" width="${bw}" height="${Math.max(0,floor-greyY)}" fill="#a7a6a1"/>`;
    bars+=`<rect x="${(x+1).toFixed(1)}" y="${redY}" width="${bw}" height="${Math.max(0,floor-redY)}" fill="#e1000f"/>`;
    labels+=`<text transform="translate(${x},${H-B+16}) rotate(-46)" text-anchor="end" font-size="10" fill="#77736d">${AGE10[i]}</text>`;
  }
  const points=target.map((p,i)=>`${px(i)},${py(p)}`).join(' ');
  $('normalChart').innerHTML=`<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Graue Bestandsklassen 2026, rote Altersklassen ${YEAR}, grüne Normalwald-Zielstruktur">${grid}${bars}<polyline points="${points}" stroke="#367b56" stroke-width="3.2" stroke-dasharray="9 7" fill="none" stroke-linecap="round" stroke-linejoin="round"/>${labels}</svg>`;
}
function similarityPlot(target){
  const values=MODEL.ageYears10.map((v,i)=>{const total=v.reduce((a,b)=>a+b,0);return {year:i+2026,n:100*(1-M.diversityIndex(v.map(x=>total?x/total:0),target))};});
  const W=850,H=180,L=48,R=20,T=16,B=33,xx=i=>L+(W-L-R)*i/(values.length-1),yy=v=>T+(H-T-B)*(100-v)/100;
  let grid='',texts='';for(let p of [0,25,50,75,100])grid+=`<line x1="${L}" x2="${W-R}" y1="${yy(p)}" y2="${yy(p)}" stroke="#e6e3de" stroke-dasharray="4 6"/><text x="${L-10}" y="${yy(p)+4}" text-anchor="end" font-size="10" fill="#837e75">${p}%</text>`;
  for(let y of [2026,2046,2066,2086,2106,2126])texts+=`<text x="${xx(y-2026)}" y="${H-9}" font-size="10" fill="#827c75" text-anchor="middle">${y}</text>`;
  const pts=values.map((v,i)=>`${xx(i)},${yy(v.n)}`).join(' '),i=YEAR-2026;
  $('similarityChart').innerHTML=`<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">${grid}${texts}<polyline points="${pts}" stroke="#e1000f" stroke-width="3.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/><line x1="${xx(i)}" x2="${xx(i)}" y1="${T}" y2="${H-B}" stroke="#b6aba0" stroke-dasharray="5 5"/><circle cx="${xx(i)}" cy="${yy(values[i].n)}" r="5.5" fill="#e1000f" stroke="white" stroke-width="2"/></svg>`;
  $('normalSimilarity').textContent=one(values[i].n)+' % Normalwald-Nähe '+YEAR;
}
function normal(){const method=$('normalMethod').value,h=+$('horizon').value,custom=GOAL;
  const target=M.normalwaldReference10(D,method,h,custom),ages=MODEL.ageYears10[YEAR-2026];
  const total=ages.reduce((a,b)=>a+b,0),shares=ages.map(x=>total?x/total:0);
  const dev=M.diversityIndex(shares,target);
  $('normalYear').textContent=YEAR;$('normalDeviation').textContent=one(dev*100)+' Pp.';
  ageNormalPlot(target);similarityPlot(target);
  $('horizonValue').textContent=h+' Jahre';$('horizonGroup').classList.toggle('hidden',method==='custom');$('customGoal').classList.toggle('hidden',method!=='custom');
  renderDeficits();
}
function draw(){if(!MODEL)return;const i=YEAR-2026,t=MODEL.years[i].n,gap=MODEL.years[TARGET_YEAR-2026].n-TARGET;
  $('kInitial').textContent=fmt(D.info.active_total);$('kProjected').textContent=fmt(t);$('kProjectedSub').textContent='Deterministisches Szenario';
  $('kTargetGap').textContent=(gap>=0?'+':'−')+fmt(Math.abs(gap));$('kTargetLabel').textContent=TARGET_YEAR;
  $('kTargetSub').textContent=`Gegenüber ${fmt(TARGET)} Bäumen im Zieljahr`;
  $('targetOutput').textContent=fmt(TARGET)+' Bäume';$('goalTopText').textContent=fmt(TARGET);
  $('targetYearOutput').textContent=TARGET_YEAR;$('chartTargetLabel').textContent=fmt(TARGET);
  $('goalStatus').textContent=`Prognose für ${TARGET_YEAR}: ${fmt(MODEL.years[TARGET_YEAR-2026].n)} Bäume · ${gap>=0?'über':'unter'} Ziel: ${fmt(Math.abs(gap))}`;
  $('kBacktest').textContent='+'+one(D.info.validation_rel_error_pct)+' %';
  $('longRangeNote').textContent=YEAR>=2050?`Fernprojektion ${YEAR}: Das Modell setzt historische Ausfallraten und die gewählte Pflanzstrategie langfristig unverändert fort. Die tatsächlichen Klima-, Standort- und Betriebsrisiken sind nicht quantifiziert.`:`Planung bis ${YEAR}: Auch kurzfristige Resultate sind Szenarien. 100-Jahre-Rechnungen sind besonders unsicher; die Bandbreite aus Monte-Carlo-Läufen berücksichtigt nicht alle Ursachen.`;
  $('replaceValue').textContent=cached.replace+' %';$('delayValue').textContent=cached.delay+' Jahre';$('annualValue').textContent=fmt(cached.annual);$('endExtraValue').textContent=cached.endExtra;
  mainChart();bars('agePreview',AGE.map((v,j)=>({name:v+' Jahre',n:MODEL.ageYears[i][j]})),'');
  const sb=Object.entries(MODEL.speciesYears[i]).map(([name,n])=>({name,n})).sort((a,b)=>b.n-a.n);bars('speciesPreview',sb.slice(0,8),'red');
  bars('mixForecast',MIX.map(m=>({name:m.s,n:MODEL.speciesYears[i][m.s]||0})).sort((a,b)=>b.n-a.n),'red');normal();
}
function makeMix(){const html=MIX.map((m,i)=>`<div class="mixRow"><label for="mixSp${i}" class="fieldNote">Baumart ${i+1}</label><select id="mixSp${i}">${species.map(s=>`<option value="${esc(s)}" ${s===m.s?'selected':''}>${esc(s)}</option>`).join('')}</select><div class="settop"><label for="mixWeight${i}">Gewichtung</label><strong id="mixShare${i}">–</strong></div><input type="range" id="mixWeight${i}" min="0" max="100" step="5" value="${m.weight}" aria-label="Pflanzanteil ${esc(m.s)}"/></div>`).join('');$('mixInputs').innerHTML=html;
MIX.forEach((_,i)=>{$('mixSp'+i).addEventListener('change',e=>{MIX[i].s=e.target.value;updateMix();laterCalc()});$('mixWeight'+i).addEventListener('input',e=>{MIX[i].weight=+e.target.value;updateMix();laterCalc()})});updateMix();}
function updateMix(){let z=MIX.reduce((a,x)=>a+Math.max(0,x.weight),0);MIX.forEach((x,i)=>{$('mixShare'+i).textContent=one(z?100*x.weight/z:100/MIX.length)+' %';});}
function goalInputs(){const root=$('goalInputs');
  root.innerHTML='<p class="fieldNote">Gewichte für die 18 Altersklassen, automatisch auf 100 % normiert.</p><div class="customGrid">'+AGE10.map((a,i)=>`<div class="customRow"><label for="goal${i}">${esc(a)} Jahre</label><input id="goal${i}" type="number" min="0" max="100" step="1" value="${GOAL[i]}" aria-label="Gewicht ${esc(a)} Jahre"/></div>`).join('')+'</div>';
  GOAL.forEach((x,i)=>{$('goal'+i).addEventListener('input',()=>{GOAL=AGE10.map((_,j)=>Math.max(0,+$('goal'+j).value||0));normal();});});
}
function solve(){const result=$('solverResult');result.classList.remove('hidden','warning');const S=settings();const r=M.findExtra(D,S,TARGET_YEAR,TARGET);if(r===null){result.classList.add('warning');result.textContent='Für das Ausgangsjahr 2026 kann eine künftige Pflanzstrategie den bereits bestehenden Bestand nicht rückwirkend verändern.';return;}
  if(r.annual==null){result.classList.add('warning');result.textContent=`Bis ${TARGET_YEAR} kann dein Ziel von ${fmt(TARGET)} Bäumen selbst mit ${fmt(r.limit)} Zusatzpflanzungen pro Jahr innerhalb des gewählten Pflanzfensters nicht erreicht werden.`;return;}
  if(r.annual>1500){result.classList.add('warning');result.innerHTML=`Rechnerisch wären etwa <b>${fmt(r.annual)} zusätzliche Pflanzungen pro Jahr</b> nötig. Der normale Regler endet bei 1’500; wir ändern ihn nicht automatisch. Dies ist <b>keine</b> Aussage zur praktischen Machbarkeit.`;return;}
  $('annual').value=r.annual;calc();result.innerHTML=`<b>${fmt(r.annual)} zusätzliche Pflanzungen pro Jahr</b> würden bei den aktuellen Testannahmen bis <b>${TARGET_YEAR}</b> ungefähr <b>${fmt(r.predicted)} Bäume</b> ergeben. Deine Zielvorgabe beträgt <b>${fmt(TARGET)} Bäume</b>. Dies ist eine deterministische Suche, keine validierte betriebliche Pflanzempfehlung.`;
}
function reset(){const defaults={replace:80,delay:2,annual:0,endExtra:2030,replMode:'same'};Object.entries(defaults).forEach(([k,v])=>$(k).value=v);MIX=first.map((s,i)=>({s:species.includes(s)?s:species[i],weight:[25,20,20,20,15][i]}));
  TARGET=17000;TARGET_YEAR=2030;$('targetInput').value=17000;$('targetSlider').value=17000;$('targetYearSlider').value=2030;
  makeMix();$('solverResult').classList.add('hidden');calc();}
function quality(){let x=D.info;const fields=[['Im Kataster aktiv',fmt(x.active_total)],['Alter ergänzt',fmt(x.imputed_ages)],['Historische Fällungen',fmt(x.felled_status_total)],['Rücktest dokumentiert',fmt(x.validation_observed)],['Rücktest erwartet',fmt(x.validation_predicted)],['Trainings-Beobachtungsjahre',fmt(x.train_tree_years)]];
$('qualityStats').innerHTML=fields.map(([k,v])=>`<div><strong>${v}</strong><span>${k}</span></div>`).join('');
$('backtestTable').innerHTML='<table class="table"><thead><tr><th>Jahr</th><th>Dokumentiert</th><th>Erwartet</th><th>Abweichung</th></tr></thead><tbody>'+D.validation.map(e=>`<tr><td>${e.year}</td><td>${fmt(e.observed)}</td><td>${fmt(e.expected)}</td><td>${e.error>=0?'+':'−'}${fmt(Math.abs(e.error))}</td></tr>`).join('')+'</tbody></table>';
$('sourceCards').innerHTML=D.external.map(s=>`<div class="sourceCard"><h3>${esc(s.city)}</h3><strong>${esc(s.status)}</strong>${s.records!=null?`<div class="records">${fmt(s.records)} Datensätze</div>`:''}<small>${esc(s.scope||'')}</small><small>${esc(s.access)}</small><a href="${esc(s.source)}" target="_blank" rel="noopener noreferrer">Datenquelle anzeigen ↗</a></div>`).join('');
}
function initZurich(){const input=$('zurichFile'),out=$('zurichCompare');input.addEventListener('change',async()=>{const file=input.files&&input.files[0];if(!file)return;try{if(file.size>20_000_000)throw new Error('Datei zu gross (max. 20 MB).');const imported=JSON.parse(await file.text());const list=imported.external;if(!Array.isArray(list))throw new Error('Keine externen Datensätze im JSON.');const z=list.find(e=>/Zürich.*Baumkataster|Zuerich.*Baumkataster/i.test(e.city||'')&&e.top_species&&e.records);if(!z)throw new Error('Zürcher Baumkataster mit top_species nicht enthalten. Bitte Start_mit_Opendata.cmd verwenden.');const rows=Object.entries(z.top_species).slice(0,10).map(([name,count])=>({name,count,local:D.species_counts[name]}));out.innerHTML='<div class="compareStatus">Vergleich anhand echter Artenhäufigkeiten. Gesamtbestand Zürich: '+fmt(z.records)+'; Winterthur: '+fmt(D.info.active_total)+'. Keine Nutzung als Ausfallrisiko.</div>'+rows.map(r=>{let zshare=100*r.count/z.records,lshare=r.local!=null?100*r.local/D.info.active_total:null;return '<div class="compareRow"><b>'+esc(r.name)+'</b><div class="compareBars"><div>Zürich '+one(zshare)+' %<div class="barbg"><i class="barfill zurich" style="display:block;width:'+Math.min(100,zshare*15)+'%"></i></div></div><div>Winterthur '+(lshare==null?'nicht einzeln ausgewiesen':one(lshare)+' %')+'<div class="barbg"><i class="barfill" style="display:block;width:'+(lshare==null?0:Math.min(100,lshare*15))+'%"></i></div></div></div></div>';}).join('');}catch(e){out.innerHTML='<div class="compareStatus" role="alert">Import nicht möglich: '+esc(e.message)+'</div>';}});}
function bind(){document.querySelectorAll('.tab').forEach(t=>t.addEventListener('click',()=>{document.querySelectorAll('.tab').forEach(b=>{let active=b===t;b.classList.toggle('active',active);b.setAttribute('aria-selected',String(active))});document.querySelectorAll('.tabbody').forEach(b=>b.classList.toggle('hidden',b.id!=='tab-'+t.dataset.tab));selectedTab=t.dataset.tab;}));
$('yearSlider').addEventListener('input',redraw);
document.querySelectorAll('[data-year-jump]').forEach(button=>button.addEventListener('click',()=>{$('yearSlider').value=button.dataset.yearJump;redraw();}));
const updateTarget=(val)=>{if(!Number.isFinite(+val)||+val<1)return;TARGET=Math.min(100000,Math.round(+val));$('targetInput').value=TARGET;$('targetSlider').value=Math.min(50000,Math.max(10000,TARGET));$('solverResult').classList.add('hidden');draw();};
$('targetInput').addEventListener('change',e=>updateTarget(e.target.value));
$('targetSlider').addEventListener('input',e=>updateTarget(e.target.value));
$('targetYearSlider').addEventListener('input',e=>{TARGET_YEAR=+e.target.value;$('solverResult').classList.add('hidden');draw();});
inputIDs.forEach(id=>$(id).addEventListener('input',()=>{$('solverResult').classList.add('hidden');laterCalc();}));
$('normalMethod').addEventListener('change',normal);$('horizon').addEventListener('input',normal);
$('deficitBasis').addEventListener('change',renderDeficits);
$('deficitExport').addEventListener('click',()=>{
  const x=getDeficitAnalysis();
  const rows=['Altersklasse;Betrachtungsjahr;Ist_Baeume;Soll_Baeume;Defizit_fehlende_Baeume;Ueberhang_Baeume;Differenz_Soll_minus_Ist;Vergleichsbasis'];
  const value=v=>String(Number(v.toFixed(2))).replace('.',',');
  x.rows.forEach((r,i)=>rows.push([AGE10[i],YEAR,value(r.current),value(r.target),value(r.deficit),value(r.surplus),value(r.difference),$('deficitBasis').value==='same'?'gleicher Bestand':'Zielbestand'].join(';')));
  download('\uFEFF'+rows.join('\r\n'),'Normalwald_Defizite_'+YEAR+'_Ziel_'+TARGET+'.csv','text/csv;charset=utf-8');
});
$('solve').addEventListener('click',solve);$('reset').addEventListener('click',reset);
$('exportJson').addEventListener('click',()=>{let obj={generated_at:new Date().toISOString(),model:'Winterthur v8.3 · deterministische Szenarioplanung (100 Jahre)',data_year:'2026-04-13',warning:'Nicht validierte Fernprojektion 2026–2126 mit konstanten empirischen Fällungsraten. Externe Baumkataster NICHT für Mortalität verwendet.',selected_year:YEAR,target_count:TARGET,target_year:TARGET_YEAR,settings:settings(),normalwald:{age_classes:AGE10,method:$('normalMethod').value,horizon:+$('horizon').value,custom_goal:GOAL,vergleichsbasis:$('deficitBasis').value},years:MODEL.years};download(JSON.stringify(obj,null,2),'Baumversorgung_Szenario_'+YEAR+'.json','application/json')});
$('exportCsv').addEventListener('click',()=>{const rows=['Jahr;Szenario_Baeume;Python_Referenz_Mittel;Referenz_P05;Referenz_P95;Ausfaelle_Erwartungswert;Ersatz_Eingesetzt;Zusatzpflanzungen'];MODEL.years.forEach((v,i)=>{let r=D.years[i],f=MODEL.flows[i];rows.push([v.year,v.n,r.mean,r.p05,r.p95,f.deaths,f.replacements,f.additional].map(x=>typeof x==='number'?String(+x.toFixed(3)).replace('.',','):x).join(';'));});download('\uFEFF'+rows.join('\r\n'),'Baumversorgung_Jahreswerte.csv','text/csv;charset=utf-8')});
}
if(!D||!M){document.body.innerHTML='<p style="padding:40px;color:red">Die Modell-Dateien konnten nicht geladen werden.</p>';return;}
makeMix();goalInputs();quality();initZurich();bind();calc();
})();
