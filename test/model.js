/* Baumversorgung Winterthur – v8 deterministischer Szenariokern.
   Fachliche Grenze: empirische dokumentierte Fällungsraten (nicht natürliche Mortalität).
   Die kalibrierte Python-Monte-Carlo-Referenz wird separat aus data.js dargestellt.
   Öffentliches Eingabematerial ist nach Baumarten und 5-Jahres-Alter aggregiert. */
(function(g){'use strict';
const B=[15,30,50,70,100], ONE=2026;
const AGE10_LABELS=[...Array.from({length:17},(_,i)=>`${i*10}–${i*10+9}`),'170+'];
const ageClass10=age=>Math.min(17,Math.max(0,Math.floor(age/10)));
const ageClass = age => age<15?0:age<30?1:age<50?2:age<70?3:age<100?4:5;
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function initCohorts(d){return d.initial.map(([s,age,n])=>({s,age,n}));}
function getMix(s){let rows=(s.mix||[]).filter(m=>m.s&&m.weight>0),z=rows.reduce((p,m)=>p+m.weight,0);if(!z)return [{s:'Weitere Baumarten',w:1}];const merged=new Map();for(let m of rows)merged.set(m.s,(merged.get(m.s)||0)+m.weight/z);return [...merged].map(([sp,w])=>({s:sp,w}));}
function run(d,settings,stopYear){const s=Object.assign({replace:80,delay:2,annual:0,endExtra:2030,replMode:'same',mix:[]},settings||{});
let c=initCohorts(d), q=new Map(),years=[],speciesYears=[],ageYears=[],ageYears10=[],flows=[];
const to=Math.min(ONE+d.years.length-1,stopYear||ONE+d.years.length-1);
const rate=(sp,age)=>d.species_rates[sp+'|'+ageClass(age)]??d.age_rates[ageClass(age)];
const mix=getMix(s);
for(let year=ONE;year<=to;year++){
let diedCount=0, addedRepl=0,addedExtra=0,waiting=0;
if(year>ONE){let incoming=q.get(year)||[];q.delete(year);addedRepl=incoming.reduce((sum,r)=>sum+r.n,0);c.push(...incoming);
if(s.annual>0&&year<=s.endExtra){addedExtra=s.annual;for(let m of mix)c.push({s:m.s,age:9,n:s.annual*m.w});}
let next=[],gone=new Map();
for(let cohort of c){let age=Math.min(250,cohort.age+1),p=rate(cohort.s,age);let deaths=cohort.n*p,surv=cohort.n-deaths;diedCount+=deaths;if(surv>1e-9)next.push({s:cohort.s,age,n:surv});if(deaths>1e-9)gone.set(cohort.s,(gone.get(cohort.s)||0)+deaths*s.replace/100);}
if(s.replace>0){let repl=[];if(s.replMode==='same')repl=[...gone].map(([sp,n])=>({s:sp,age:9,n}));else {let sum=[...gone.values()].reduce((a,b)=>a+b,0);repl=mix.map(m=>({s:m.s,age:9,n:sum*m.w}));}
if(s.delay===0){addedRepl+=repl.reduce((a,t)=>a+t.n,0);next.push(...repl.map(t=>({...t,age:10})));}else{let due=year+s.delay;q.set(due,[...(q.get(due)||[]),...repl]);}}
c=next;
}
let sp=new Map(),age=Array(6).fill(0),age10=Array(18).fill(0),total=0;for(let t of c){sp.set(t.s,(sp.get(t.s)||0)+t.n);age[ageClass(t.age)]+=t.n;age10[ageClass10(t.age)]+=t.n;total+=t.n;}
waiting=[...q.values()].flat().reduce((a,t)=>a+t.n,0);
years.push({year,n:total});speciesYears.push(Object.fromEntries(sp));ageYears.push(age);ageYears10.push(age10);flows.push({year,deaths:diedCount,replacements:addedRepl,additional:addedExtra,queued:waiting});
}return {years,speciesYears,ageYears,ageYears10,flows,settings:s};}
// This is an explicit *target*, not an empirically optimum city-tree age structure.
function normalwaldReference(d,method,horizon,custom){const ranges=[[0,15],[15,30],[30,50],[50,70],[70,100],[100,Infinity]],out=Array(6).fill(0),H=clamp(Math.round(horizon),80,200);
if(method==='custom'){let a=custom.map(x=>Math.max(0,+x||0)),sum=a.reduce((x,y)=>x+y,0);return sum>0?a.map(n=>n/sum):[1/6,1/6,1/6,1/6,1/6,1/6];}
if(method==='equal') {for(let i=0;i<6;i++)out[i]=Math.max(0,Math.min(ranges[i][1],H)-ranges[i][0]);}
else{let survival=1;for(let age=0;age<H;age++){out[ageClass(age)]+=survival;survival*=1-d.age_rates[ageClass(age)];}}
let z=out.reduce((a,b)=>a+b,0);return out.map(v=>v/z);}
// 10-Jahres-Struktur aus den aggregierten Alterskohorten; kein exakter Einzelbaum-Kataster.
function normalwaldReference10(d,method,horizon,custom){
  const out=Array(18).fill(0),H=clamp(Math.round(horizon),80,180);
  if(method==='custom'){
    const v=Array.from({length:18},(_,i)=>Math.max(0,Number(custom[i])||0)),z=v.reduce((a,b)=>a+b,0);
    return z>0?v.map(x=>x/z):Array.from({length:18},(_,i)=>i<12?1/12:0);
  }
  let surviving=1;
  for(let age=0;age<H;age++){
    out[ageClass10(age)]+=method==='equal'?1:surviving;
    if(method!=='equal')surviving*=1-d.age_rates[ageClass(age)];
  }
  const z=out.reduce((a,b)=>a+b,0);
  return out.map(x=>z?x/z:0);
}
function diversityIndex(shares,target){return .5*shares.reduce((a,x,i)=>a+Math.abs(x-target[i]),0);}
// Altersklassen-Lücken zum bewusst gewählten Planungsziel (kein Pflanzbedarf).
// Vergleich: gleicher Gesamtbestand oder expliziter Zielbestand; keine Mortalitätsschätzung.
function normalwaldDeficits(ages,target,comparisonTotal){
  if(ages.length!==target.length)throw new Error('Altersklassen von Bestand und Ziel passen nicht zusammen.');
  const current=ages.reduce((a,n)=>a+n,0);
  const goal=Math.max(0,Number(comparisonTotal)||0);
  const rows=ages.map((actual,i)=>{
    const planned=goal*target[i],difference=planned-actual;
    return {index:i,current:actual,target:planned,difference,
      deficit:Math.max(0,difference),surplus:Math.max(0,-difference),
      shareActual:current?actual/current:0,shareTarget:target[i]};
  });
  const missing=rows.reduce((s,r)=>s+r.deficit,0),excess=rows.reduce((s,r)=>s+r.surplus,0);
  return {rows,missing,excess,balance:missing-excess,referenceTotal:goal,currentTotal:current,
    largestMissing:rows.reduce((a,r)=>r.deficit>a.deficit?r:a,rows[0]),
    largestExcess:rows.reduce((a,r)=>r.surplus>a.surplus?r:a,rows[0])};
}

function findExtra(d,s,targetYear,targetCount,maxExtra=5000){if(targetYear<=ONE)return null;
const fn=annual=>run(d,{...s,annual},targetYear).years.at(-1).n;
let lo=0,hi=1,baseline=fn(0);if(baseline>=targetCount)return {annual:0,predicted:baseline};
while(hi<maxExtra&&fn(hi)<targetCount)hi*=2;hi=Math.min(maxExtra,hi);
if(fn(hi)<targetCount)return {annual:null,predicted:fn(hi),limit:hi};
while(hi-lo>1){let mid=Math.floor((lo+hi)/2);if(fn(mid)>=targetCount)hi=mid;else lo=mid;}
return {annual:hi,predicted:fn(hi)};
}
g.BaumModel={run,normalwaldReference,normalwaldReference10,normalwaldDeficits,diversityIndex,findExtra,ageClass,ageClass10,AGE10_LABELS,getMix};
})(window);
