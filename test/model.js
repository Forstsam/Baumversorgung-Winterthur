"use strict";
const START=DATA.startCount,A=DATA.age,H=DATA.haz,MAX=200,BASEYEAR=2026;
const species=[...DATA.species.map(s=>({...s})),{name:'__other__',label:'Übrige / nicht zugeordnet',count:Math.max(0,DATA.startCount-DATA.species.reduce((n,s)=>n+s.count,0)),life:null,risk:1,fit:{ssp126:null,ssp370:null,ssp585:null}}];
const byName=new Map(species.map(s=>[s.name,s]));
const speciesIndex=new Map(species.map((s,i)=>[s.name,i]));
const clamp=(n,lo,hi)=>Math.max(lo,Math.min(hi,n));const sum=a=>a.reduce((x,y)=>x+y,0);const fmt=n=>Math.round(n).toLocaleString('de-CH').replace(/\u00a0/g,'’');const percent=n=>(100*n).toFixed(1)+' %'; const el=id=>document.getElementById(id);const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const baseParams={year:2030,goal:17000,plantings:300,start:2027,end:2030,replace:.9,delay:1,est:1,scenario:'ssp126',stress:1,ref:'survival',life:120,priority:'balance'};
let p={...baseParams},viewYear=2050,tab='ziel',mixMode='observed',riskMode='neutral',suggested=null,runCache={key:null,results:null};
let mixes=[{name:'Quercus robur',share:40},{name:'Acer campestre',share:30},{name:'Tilia cordata',share:20},{name:'Carpinus betulus',share:10}];
function classes(a){const z=Array(18).fill(0);for(let i=0;i<a.length;i++)z[Math.min(17,Math.floor(i/10))]+=a[i];return z;}
function similarity(a,target){const total=sum(a);return total<=0?0:100*(1-.5*a.reduce((r,v,i)=>r+Math.abs(v/total-target[i]),0));}
function reference(){const lifespan=p.life,values=Array(MAX).fill(0);for(let i=0;i<Math.min(MAX,lifespan);i++){if(p.ref==='equal')values[i]=1;else values[i]=i===0?1:values[i-1]*(1-clamp(H[i-1],0,.5));}const cls=classes(values);return cls.map(v=>v/sum(cls));}
function observedMix(){return species.filter(s=>s.count>0).map(s=>({name:s.name,weight:s.count/START}));}
function activeMix(){if(mixMode==='observed')return observedMix();const tot=mixes.reduce((v,s)=>v+s.share,0);return mixes.filter(s=>s.share>0&&byName.has(s.name)).map(s=>({name:s.name,weight:s.share/tot}));}
function speciesFactor(s,sc){if(!s||s.name==='__other__')return 1;const v=s.fit[sc];const fit=Number.isFinite(v)?v:.55;return clamp((s.risk||1)*Math.pow(130/clamp(s.life||130,60,260),.34)*(1+.38*(1-fit)),.45,1.7);}
// Annual cohort-based expectation model. All category populations always sum to the simulated stock.
// Deterministisches Erwartungswert-Modell; nur bis zum tatsächlich benötigten Jahr rechnen.
// Für jedes Jahr werden Ausfälle und Pflanzungen in kohortenbasierten Bestandsbilanzen geführt.
function simulate(params,risk,mixArr,until=2126){
  const ageOld=Array(MAX).fill(0);
  const scale=START/sum(A);
  for(let i=0;i<A.length;i++) ageOld[i]=A[i]*scale;
  let old=ageOld,cohorts=[],prevDeaths=[0],path=[];
  const shares=species.map(s=>s.count/START);
  // Zuordnung einmal vorberechnen: kein findIndex bei jeder Art und jedem Jahr.
  const mix=mixArr.filter(z=>z.weight>0).map(z=>({
    index:speciesIndex.get(z.name),weight:z.weight,
    factor:risk==='proxy'?speciesFactor(byName.get(z.name),params.scenario):1
  })).filter(z=>z.index!==undefined);
  const maxYear=Math.max(BASEYEAR,Math.min(2126,Math.round(until)));
  for(let y=BASEYEAR;y<=maxYear;y++){
    let deaths=0,extra=0,replacement=0;
    if(y>BASEYEAR){
      const next=Array(MAX).fill(0),t=y-BASEYEAR;
      const cm=risk==='proxy'?({ssp126:1,ssp370:1.02,ssp585:1.085}[params.scenario]||1)*(1+.35*t/100):1;
      const mult=cm*params.stress;
      for(let age=0;age<MAX;age++){
        const gone=old[age]*clamp(H[age]*mult,.0001,.5);
        deaths+=gone;
        next[Math.min(MAX-1,age+1)]+=old[age]-gone;
      }
      old=next;
      for(const cohort of cohorts){
        const gone=cohort.n*clamp(H[cohort.age]*mult*cohort.factor,.0001,.5);
        deaths+=gone;cohort.n-=gone;cohort.age=Math.min(MAX-1,cohort.age+1);
      }
      // Verzögerung 1 = Ersatz im Folgejahr; Verzögerung 0 = gleiches Jahr.
      const eligible=params.delay===0?deaths:(t-params.delay>=1?prevDeaths[t-params.delay]||0:0);
      replacement=eligible*params.replace;
      extra=y>=params.start&&y<=params.end?params.plantings:0;
      const planted=(extra+replacement)*params.est;
      for(const z of mix){if(planted>0)cohorts.push({age:10,n:planted*z.weight,index:z.index,factor:z.factor});}
      prevDeaths[t]=deaths;
    }
    const initialRemaining=sum(old),tally=shares.map(s=>s*initialRemaining),hist=classes(old);
    for(const cohort of cohorts){
      tally[cohort.index]+=cohort.n;
      hist[Math.min(17,Math.floor(cohort.age/10))]+=cohort.n;
    }
    const total=sum(tally),histTotal=sum(hist);
    if(!Number.isFinite(total)||total<0||Math.abs(total-histTotal)>Math.max(.005,total*1e-9))
      throw new Error('Bestandsbilanz ungültig: '+y+' '+(total-histTotal));
    if(cohorts.some(z=>z.n < -1e-8))throw new Error('Negative Pflanzkohorte im Jahr '+y);
    path.push({year:y,total,hist,deaths,extra,replacement,planting:extra+replacement,composition:tally});
  }
  return path;
}
// Nur Modellparameter in den Cache-Schlüssel aufnehmen. Das gewählte Anzeigejahr,
// Bestandsziel und die Referenzlinie verändern keinen Simulationslauf.
const BASELINE=simulate(baseParams,'neutral',observedMix());
function getRuns(){
  const key=JSON.stringify({plantings:p.plantings,start:p.start,end:p.end,
    replace:p.replace,delay:p.delay,est:p.est,scenario:p.scenario,stress:p.stress,
    riskMode,mixMode,mixes:mixMode==='custom'?mixes:null});
  if(runCache.key===key)return runCache.results;
  const custom=simulate(p,riskMode,activeMix());
  runCache={key,results:{base:BASELINE,custom}};
  return runCache.results;
}
function invalidate(){suggested=null;}

// Kleine automatische Qualitätssicherung ohne externe Testbibliothek.
function runModelChecks(){
  const checks=[];
  const check=(label,test)=>{try{checks.push({label,pass:!!test()});}catch(e){checks.push({label:label+' ('+e.message+')',pass:false});}};
  check('Kataster-Gesamtbestand und Artensumme stimmen überein',()=>sum(species.map(s=>s.count))===START);
  check('Pflanzmix gewichtet zu insgesamt 100 %',()=>Math.abs(sum(activeMix().map(z=>z.weight))-1)<1e-9);
  const path=simulate(p,riskMode,activeMix(),Math.min(2126,Math.max(p.year,2050)));
  check('Nichtnegative Bestände und Jahrgänge',()=>path.every(z=>z.total>=0&&z.hist.every(v=>v>=0)&&z.composition.every(v=>v>=0)));
  check('Für jedes Jahr gleiche Summen aus Altersstruktur und Arten',()=>path.every(z=>Math.abs(z.total-sum(z.hist))<0.005&&Math.abs(z.total-sum(z.composition))<0.005));
  check('Jährliche Massenbilanz: Bestand = Vorjahr − Ausfälle + wirksame Pflanzungen',()=>path.slice(1).every((z,i)=>Math.abs(z.total-(path[i].total-z.deaths+(z.extra+z.replacement)*p.est))<0.005));
  check('Ausgangsbestand 2026 unverändert',()=>Math.abs(path[0].total-START)<0.005);
  return checks;
}
