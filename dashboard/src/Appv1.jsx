import React, { useMemo, useState } from "react";
import { ComposedChart, Area, Line, XAxis, YAxis, ReferenceLine, ReferenceDot, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

/* ─────────────────────────────────────────────────────────────
   Winterthur Stadtbaum-Modell v7 — interaktives Dashboard
   Repliziert die Engine aus winterthur_tree_stochastic_goal_planning_v7.py:
   p_fail = clip(base_p(Alter) · stress · (1+trend) · site^1 · mgmt^0.6 · life_mult, 1e-4, 0.50)
   base_p = empirisch kalibrierte Alters-Hazard aus dem Kataster (Sterbetafel,
   Laplace α=0.5, min_risk_set=20, clip≤0.35). Neue Bäume starten mit Alter 10.
   ───────────────────────────────────────────────────────────── */

/* Winterthur-Identität: Schwarz-Rot auf Weiss (Stadtlogo), heraldisches Rot, Schweizer Raster */
const PINE="#ece9e1",PANEL="#ffffff",CARD="#ffffff",LINE_GRID="#d9d7cf";
const PAPER="#1c1c1a",SAGE="#6f6d66",MOSS="#e1000f",MOSS_DIM="#b51020";
const AMBER="#c8860d",BARK="#9a9890",BRICK="#b00020",TEXT_MUTE="#7a786f";
const GREEN="#2f7d4f";
const CURRENT_YEAR=2026,YEARS=100,MAXAGE=200,MILESTONES=[4,25,50,100];

/* Echte kalibrierte Alters-Hazard-Kurve (Index = Alter, aus Kataster 2026) */
const HAZARD=[0.00136,0.00238,0.00443,0.00581,0.00651,0.00489,0.00659,0.00482,0.00448,0.0065,0.00457,0.00865,0.00868,0.00454,0.00495,0.00626,0.00629,0.00477,0.00548,0.00632,0.00696,0.00641,0.00757,0.00548,0.00757,0.00466,0.00666,0.00732,0.00787,0.01058,0.01247,0.00691,0.00887,0.0072,0.01138,0.01162,0.01116,0.01002,0.00865,0.01005,0.0141,0.00958,0.01369,0.00938,0.00864,0.00923,0.01028,0.01451,0.01048,0.01583,0.01571,0.01248,0.01266,0.0127,0.01326,0.01017,0.01059,0.01388,0.01328,0.00888,0.01312,0.0113,0.02701,0.01075,0.01403,0.01136,0.00874,0.00972,0.01031,0.00672,0.01377,0.00798,0.01003,0.0107,0.0085,0.01178,0.01277,0.00899,0.00508,0.01437,0.01654,0.01495,0.00816,0.00928,0.00655,0.01255,0.00792,0.0062,0.00867,0.00841,0.01725,0.00886,0.00927,0.00672,0.00319,0.00963,0.00923,0.00624,0.01123,0.00927,0.0054,0.01355,0.00467,0.00722,0.01234,0.00674,0.00619,0.00615,0.00694,0.0131,0.01103,0.00581,0.00668,0.01168,0.01019,0.00702,0.00624,0.01197,0.00583,0.00483,0.00812,0.01304,0.00523,0.0041,0.00648,0.00178,0.01029,0.00198,0.01525,0.00482,0.01042,0.00492,0.01686,0.01274,0.00836,0.01767,0.00235,0.01402,0.02376,0.00877,0.00884,0.00495,0.00497,0.007,0.00706,0.00507,0.00714,0.00598,0.0012,0.01095,0.01351,0.01122,0.00897,0.00646,0.00649,0.01197,0.00672,0.00177,0.01311,0.00189,0.0407,0.01008,0.35,0.02119,0.01327,0.00446,0.03125,0.01389,0.07944,0.01648,0.00556,0.00556,0.00562,0.00568,0.00568,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705,0.01705];
const AGE_HIST=[75,270,333,724,649,479,222,304,218,205,186,258,542,243,279,301,543,327,395,243,407,285,189,150,88,177,209,116,146,194,120,240,95,71,195,199,396,136,97,44,250,393,39,45,75,56,276,64,57,66,148,178,90,64,129,11,461,10,148,27,139,138,24,31,35,29,471,18,127,40,50,105,18,14,101,9,321,8,65,9,18,122,2,4,59,0,198,6,20,13,51,85,36,110,5,2,148,5,27,6,15,108,6,2,3,14,146,7,24,4,11,10,25,0,5,1,239,6,7,4,36,10,3,2,2,15,60,4,17,3,1,28,4,1,2,2,105,2,1,0,0,0,1,1,0,1,71,0,5,0];
const START_ACTIVE=16621, AVG_HAZARD_TODAY=0.0081, LIFE_REF=130;

/* Pflanzstrategie: Näherung des Speziesauswahl-Effekts als Hazard-Faktor neuer Bäume */
const STRAT_NEW={ same_mix:1.0, long_life:0.90, climate_fit:0.93, balanced:0.91 };
const STRAT_LABEL={ same_mix:"Status quo", long_life:"Langlebig", climate_fit:"Klima-Fit", balanced:"Ausgewogen" };

/* Bestandsgewichteter climate_mult aus TreeGOER-Exceedance (future bio05 vs bio05_q-Grenze der Art),
   exp(-tree_k·max(0,future−grenze)), 1/clip(.,0.2,2.0), gemittelt über aktive Bäume. 45 % Artabdeckung. */
const CLIMATE={
  ssp126:{ q95:{0.08:1.000,0.15:1.000}, qrt3:{0.08:1.013,0.15:1.026} },
  ssp370:{ q95:{0.08:1.001,0.15:1.001}, qrt3:{0.08:1.015,0.15:1.030} },
  ssp585:{ q95:{0.08:1.033,0.15:1.076}, qrt3:{0.08:1.095,0.15:1.229} },
};
const SCEN_LABEL={ ssp126:"SSP1-2.6 · 26.9°C", ssp370:"SSP3-7.0 · 27.1°C", ssp585:"SSP5-8.5 · 32.2°C" };

const DEFAULTS={
  N0:START_ACTIVE, scenario:"ssp126", treeQ:"q95", treeK:0.08, manualStress:1.0, siteFactor:1.0, mgmtFactor:1.0, climateTrendEnd:0.35,
  replacementRate:0.8, replacementDelay:2, annualNewTrees:300, newStart:1, newEnd:4, initAge:10,
  strategy:"balanced", lifeHazardWeight:0.5, lifeMode:"per_run",
  nRuns:100, targetCount:17000, seed:42, viewYearOff:24,
};
const PRESETS={
  konservativ:{annualNewTrees:250,strategy:"balanced",replacementRate:0.8},
  mittel:{annualNewTrees:300,strategy:"balanced",replacementRate:0.8},
  robust:{annualNewTrees:350,strategy:"balanced",replacementRate:0.8},
  maximal:{annualNewTrees:300,strategy:"balanced",replacementRate:1.0},
};

function mulberry32(a){return function(){a|=0;a=(a+0x6d2b79f5)|0;let t=Math.imul(a^(a>>>15),1|a);t=(t+Math.imul(t^(t>>>7),61|t))^t;return((t^(t>>>14))>>>0)/4294967296;};}
function makeNormal(rng){let sp=null;return()=>{if(sp!==null){const s=sp;sp=null;return s;}let u,v,s;do{u=rng()*2-1;v=rng()*2-1;s=u*u+v*v;}while(s>=1||s===0);const m=Math.sqrt(-2*Math.log(s)/s);sp=v*m;return u*m;};}
const haz=a=>HAZARD[Math.min(a,MAXAGE-1)];

function startCohorts(N0){const arr=new Float64Array(MAXAGE);let s=0;for(let a=0;a<AGE_HIST.length&&a<MAXAGE;a++){arr[a]=AGE_HIST[a];s+=AGE_HIST[a];}const k=N0/s;for(let a=0;a<MAXAGE;a++)arr[a]*=k;return arr;}

function runCohort(p,stochastic,normal,m){
  const stratNew=STRAT_NEW[p.strategy];
  const climateMult=(CLIMATE[p.scenario]?.[p.treeQ]?.[p.treeK])??1.0;
  const stress=climateMult*p.manualStress;
  const old=startCohorts(p.N0), neu=new Float64Array(MAXAGE);
  const ia=Math.min(Math.max(0,p.initAge),MAXAGE-1);
  const deaths=new Float64Array(YEARS+1);
  const total=new Array(YEARS+1), classOld=[],classNew=[],medAge=[],young=[],old60=[];
  const snap=(oa,na)=>{const co=Array(18).fill(0),cn=Array(18).fill(0);let tot=0,y=0,o=0;
    for(let a=0;a<MAXAGE;a++){const c=Math.min(17,Math.floor(a/10));co[c]+=oa[a];cn[c]+=na[a];const t=oa[a]+na[a];tot+=t;if(a<20)y+=t;if(a>=60)o+=t;}
    classOld.push(co);classNew.push(cn);
    let acc=0,med=0;for(let a=0;a<MAXAGE;a++){acc+=oa[a]+na[a];if(acc>=tot/2){med=a;break;}}
    medAge.push(med);young.push(tot>0?y/tot:0);old60.push(tot>0?o/tot:0);return tot;};
  total[0]=snap(old,neu);
  for(let t=1;t<=YEARS;t++){
    const trend=p.climateTrendEnd*(t/YEARS);
    const rmO=m*stress*(1+trend)*Math.pow(p.siteFactor,1.0)*Math.pow(p.mgmtFactor,0.6);
    const rmN=rmO*stratNew;
    let dTot=0;const noO=new Float64Array(MAXAGE),noN=new Float64Array(MAXAGE);
    for(let a=MAXAGE-1;a>=0;a--){
      const bp=HAZARD[a];
      let pO=bp*rmO; if(pO<0.0001)pO=0.0001; if(pO>0.50)pO=0.50;
      let pN=bp*rmN; if(pN<0.0001)pN=0.0001; if(pN>0.50)pN=0.50;
      let dO=old[a]*pO,dN=neu[a]*pN;
      if(stochastic){
        if(old[a]>0){const sd=Math.sqrt(old[a]*pO*(1-pO));dO=Math.max(0,Math.min(old[a],dO+sd*normal()));}
        if(neu[a]>0){const sd=Math.sqrt(neu[a]*pN*(1-pN));dN=Math.max(0,Math.min(neu[a],dN+sd*normal()));}
      }
      dTot+=dO+dN; if(a+1<MAXAGE){noO[a+1]=old[a]-dO;noN[a+1]=neu[a]-dN;}
    }
    deaths[t]=dTot;
    const repl=t-p.replacementDelay>=1?Math.round(p.replacementRate*deaths[t-p.replacementDelay]):0;
    const extra=(t>=p.newStart&&t<=p.newEnd)?p.annualNewTrees:0;
    noN[ia]+=repl+extra;
    for(let a=0;a<MAXAGE;a++){old[a]=noO[a];neu[a]=noN[a];}
    total[t]=snap(old,neu);
  }
  return {total,classOld,classNew,medAge,young,old60,deaths};
}

function simulate(p){
  const det=runCohort(p,false,null,1);
  const rng=mulberry32(p.seed>>>0),normal=makeNormal(rng);
  const runs=[];
  for(let r=0;r<p.nRuns;r++){
    let sd=0.06; if(p.lifeMode==="per_run") sd=Math.sqrt(sd*sd+Math.pow(0.10*p.lifeHazardWeight,2));
    let m=Math.exp(normal()*sd); m=Math.min(2.0,Math.max(0.5,m));
    runs.push(runCohort(p,true,normal,m).total);
  }
  const series=[];
  for(let t=0;t<=YEARS;t++){
    const v=runs.map(s=>s[t]).sort((a,b)=>a-b);
    const q=x=>v[Math.min(v.length-1,Math.round(x*(v.length-1)))];
    series.push({offset:t,year:CURRENT_YEAR+t,p05:q(.05),p25:q(.25),p50:q(.5),p75:q(.75),p95:q(.95),
      mean:v.reduce((a,b)=>a+b,0)/v.length,band90:[q(.05),q(.95)],band50:[q(.25),q(.75)]});
  }
  return {series,det};
}

const fmt=n=>Math.round(n).toLocaleString("de-CH").replace(/,/g,"\u2019");
const yearAt=o=>CURRENT_YEAR+o;
function verdict(pt,target){if(!pt)return{label:"—",color:TEXT_MUTE,note:""};
  if(pt.p05>=target)return{label:"Robust erreicht",color:GREEN,note:"95 % der Läufe über dem Ziel"};
  if(pt.p50>=target)return{label:"Wahrscheinlich erreicht",color:"#5a9e63",note:"in über der Hälfte der Läufe"};
  if(pt.mean>=target)return{label:"Knapp / im Mittel",color:AMBER,note:"Mittelwert über Ziel, Median darunter"};
  return{label:"Verfehlt",color:BRICK,note:"Ziel im Median nicht erreicht"};}
function ageColor(i){const r=i/17,g=[201,199,191],a=[74,72,67];return `rgb(${Math.round(g[0]+(a[0]-g[0])*r)},${Math.round(g[1]+(a[1]-g[1])*r)},${Math.round(g[2]+(a[2]-g[2])*r)})`;}

function Slider({label,hint,value,min,max,step,onChange,display}){
  return(<div style={{marginBottom:15}}>
    <div className="flex items-baseline justify-between" style={{marginBottom:5}}>
      <span style={{fontSize:13,color:PAPER}}>{label}</span>
      <span style={{fontSize:13,color:MOSS,fontFamily:"var(--mono)",fontWeight:600}}>{display??value}</span></div>
    <input type="range" min={min} max={max} step={step} value={value} onChange={e=>onChange(parseFloat(e.target.value))} style={{width:"100%",accentColor:MOSS,height:4,cursor:"pointer"}}/>
    {hint&&<div style={{fontSize:11,color:TEXT_MUTE,marginTop:3}}>{hint}</div>}</div>);
}
function Segmented({label,value,options,onChange}){
  return(<div style={{marginBottom:15}}>
    <div style={{fontSize:13,color:PAPER,marginBottom:6}}>{label}</div>
    <div className="flex flex-wrap" style={{gap:6}}>
      {options.map(o=>{const a=o.value===value;return(<button key={o.value} onClick={()=>onChange(o.value)}
        style={{fontSize:12,padding:"6px 11px",borderRadius:7,cursor:"pointer",border:`1px solid ${a?MOSS:LINE_GRID}`,
        background:a?"rgba(225,0,15,0.07)":"transparent",color:a?MOSS:SAGE,fontFamily:"var(--mono)"}}>{o.label}</button>);})}</div></div>);
}
function Section({title,children}){return(<div style={{marginBottom:20}}>
  <div style={{fontSize:11,letterSpacing:1.6,textTransform:"uppercase",color:MOSS_DIM,fontWeight:700,marginBottom:11,borderBottom:`1px solid ${LINE_GRID}`,paddingBottom:6}}>{title}</div>{children}</div>);}
function MilestoneCard({off,pt,target}){const ok=pt.p50>=target;
  return(<div style={{background:CARD,border:`1px solid ${LINE_GRID}`,borderRadius:12,padding:"13px 15px",flex:"1 1 0",minWidth:120}}>
    <div style={{fontSize:11,color:TEXT_MUTE,letterSpacing:1}}>{yearAt(off)}</div>
    <div style={{fontFamily:"var(--display)",fontSize:28,fontWeight:600,lineHeight:1.1,color:PAPER,fontVariantNumeric:"tabular-nums"}}>{fmt(pt.p50)}</div>
    <div style={{fontSize:11,color:TEXT_MUTE,fontFamily:"var(--mono)",marginTop:2}}>{fmt(pt.p05)} – {fmt(pt.p95)}</div>
    <div style={{marginTop:7,fontSize:11,fontWeight:600,color:ok?GREEN:(off===4?BRICK:TEXT_MUTE)}}>{off===4?(ok?"Ziel erreicht":"unter Ziel"):(ok?"über Start":"unter Start")}</div></div>);}
function ChartTip({active,payload,target}){if(!active||!payload||!payload.length)return null;const d=payload[0].payload;
  return(<div style={{background:"#ffffff",border:`1px solid ${LINE_GRID}`,borderRadius:9,padding:"10px 12px",fontFamily:"var(--mono)",fontSize:12,color:PAPER}}>
    <div style={{color:MOSS,fontWeight:700,marginBottom:4}}>{d.year}</div><div>Median&nbsp;&nbsp;<b>{fmt(d.p50)}</b></div>
    <div style={{color:SAGE}}>50 %&nbsp;&nbsp;{fmt(d.p25)} – {fmt(d.p75)}</div>
    <div style={{color:TEXT_MUTE}}>90 %&nbsp;&nbsp;{fmt(d.p05)} – {fmt(d.p95)}</div>
    <div style={{color:AMBER,marginTop:3}}>Ziel&nbsp;&nbsp;{fmt(target)}</div></div>);}

function HazardSpark(){
  const W=260,Hh=46,N=151,mx=0.04;
  const pts=Array.from({length:N},(_,a)=>{const x=a/(N-1)*W;const y=Hh-Math.min(haz(a),mx)/mx*Hh;return `${x.toFixed(1)},${y.toFixed(1)}`;}).join(" ");
  return(<svg width="100%" viewBox={`0 0 ${W} ${Hh}`} preserveAspectRatio="none" style={{display:"block"}}>
    <polyline points={pts} fill="none" stroke={MOSS} strokeWidth="1.6"/>
    {[0,50,100,150].map(a=><line key={a} x1={a/(N-1)*W} y1="0" x2={a/(N-1)*W} y2={Hh} stroke={LINE_GRID} strokeDasharray="1 4"/>)}
  </svg>);
}
function Pyramid({classOld,classNew}){
  const maxV=Math.max(1,...classOld.map((v,i)=>v+classNew[i]));
  const labels=Array.from({length:18},(_,i)=>i===17?"170+":`${i*10}–${i*10+9}`);
  return(<div style={{display:"flex",flexDirection:"column-reverse",gap:3}}>
    {labels.map((lab,i)=>{const o=classOld[i],n=classNew[i],tot=o+n;
      return(<div key={i} className="flex items-center" style={{gap:8}}>
        <div style={{width:54,textAlign:"right",fontSize:10.5,color:TEXT_MUTE,fontFamily:"var(--mono)"}}>{lab}</div>
        <div style={{flex:1,height:15,background:"#f2f1ec",borderRadius:3,overflow:"hidden",display:"flex"}}>
          <div style={{width:`${o/maxV*100}%`,background:ageColor(i),transition:"width .25s"}}/>
          <div style={{width:`${n/maxV*100}%`,background:MOSS,opacity:.55,transition:"width .25s"}}/></div>
        <div style={{width:48,fontSize:10.5,color:tot>0?SAGE:LINE_GRID,fontFamily:"var(--mono)",textAlign:"right"}}>{fmt(tot)}</div></div>);})}
    <div className="flex items-center" style={{gap:8,marginBottom:4}}>
      <div style={{width:54,textAlign:"right",fontSize:10,color:MOSS_DIM,letterSpacing:1}}>ALTER</div>
      <div style={{flex:1,fontSize:10,color:MOSS_DIM,letterSpacing:1}}>ANZAHL BÄUME →</div><div style={{width:48}}/></div></div>);
}

export default function App(){
  const [p,setP]=useState(DEFAULTS);
  const [view,setView]=useState("stock");
  const set=k=>v=>setP(s=>({...s,[k]:v}));
  const preset=n=>setP(s=>({...s,...PRESETS[n]}));
  const {series,det}=useMemo(()=>simulate(p),[p]);
  const yMax=useMemo(()=>Math.max(p.targetCount,...series.map(d=>d.p95))*1.06,[series,p.targetCount]);
  const ms=MILESTONES.map(off=>({off,pt:series[off]}));
  const t2030=series[4],v=verdict(t2030,p.targetCount),endPt=series[YEARS];
  const trendEnd=p.climateTrendEnd;
  const climateMult=(CLIMATE[p.scenario]?.[p.treeQ]?.[p.treeK])??1.0;
  const stress=climateMult*p.manualStress;
  const hToday=AVG_HAZARD_TODAY*stress*Math.pow(p.siteFactor,1)*Math.pow(p.mgmtFactor,0.6);
  const hEnd=hToday*(1+trendEnd);
  const deathsY1=Math.round(det.deaths[1]);
  const yOff=p.viewYearOff;
  const ageLine=useMemo(()=>det.medAge.map((mm,i)=>({year:CURRENT_YEAR+i,med:mm,young:Math.round(det.young[i]*100),old:Math.round(det.old60[i]*100)})),[det]);
  const renew=(()=>{const o=det.classOld[yOff].reduce((a,b)=>a+b,0),n=det.classNew[yOff].reduce((a,b)=>a+b,0);return n/(o+n||1);})();

  return(<div style={{"--display":"'Inter',system-ui,sans-serif","--mono":"'JetBrains Mono',ui-monospace,monospace",
    background:PINE,minHeight:"100vh",color:PAPER,fontFamily:"'Inter',system-ui,sans-serif",padding:"clamp(16px,3vw,34px)"}}>
    <style>{`@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
      input[type=range]{-webkit-appearance:none;appearance:none;background:${LINE_GRID};border-radius:99px;}
      input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:${MOSS};cursor:pointer;border:2px solid #fff;box-shadow:0 0 0 1px ${LINE_GRID};}
      input[type=range]::-moz-range-thumb{width:14px;height:14px;border-radius:50%;background:${MOSS};cursor:pointer;border:2px solid #fff;}`}</style>
    <div style={{maxWidth:1240,margin:"0 auto"}}>
      <div style={{marginBottom:18}}>
        <div className="flex items-center" style={{gap:12,marginBottom:8}}>
          <svg width="30" height="35" viewBox="0 0 30 35" aria-hidden="true">
            <path d="M1 1 H29 V22 C29 29 22 33 15 34 C8 33 1 29 1 22 Z" fill="#fff" stroke={PAPER} strokeWidth="1.4"/>
            {[8.5,19.5].map((cx,i)=>(<g key={i} fill={MOSS}>
              <ellipse cx={cx} cy="13" rx="3.1" ry="4"/>
              <circle cx={cx+(i?-2.6:2.6)} cy="9.2" r="1.7"/>
              <rect x={cx-1} y="16" width="2" height="7" rx="1"/>
              <path d={`M${cx+(i?-3:3)} 11 q${i?-3:3} 1 ${i?-2:2} 5`} stroke={MOSS} strokeWidth="1.4" fill="none"/>
            </g>))}
          </svg>
          <div style={{fontSize:11,letterSpacing:2.5,textTransform:"uppercase",color:MOSS_DIM,fontWeight:700}}>Stadt Winterthur · Stadtbaum-Modell v7</div>
        </div>
        <h1 style={{fontFamily:"var(--display)",fontSize:"clamp(28px,4.4vw,52px)",fontWeight:800,lineHeight:1.0,margin:"0 0 8px",letterSpacing:-1.2}}>
          {view==="stock"?"Wie entwickelt sich der Baumbestand?":"Wie verschiebt sich die Altersstruktur?"}</h1>
        <div style={{width:48,height:3,background:MOSS,marginBottom:10}}/>
        <p style={{color:TEXT_MUTE,maxWidth:680,fontSize:14,lineHeight:1.5}}>
          {view==="stock"
            ?`Basis-Ausfall = kalibrierte Alters-Hazard aus dem Kataster (≈135 Fällungen/Jahr im Ist-Zustand). Ziel ${fmt(p.targetCount)} bis 2030.`
            :"Echte Altersverteilung aus den Pflanzjahren, fortgeschrieben mit der kalibrierten Alters-Hazard. Neue Bäume starten mit Alter "+p.initAge+"."}</p></div>

      <div className="flex" style={{gap:8,marginBottom:18}}>
        {[["stock","Bestand"],["demography","Demografie"]].map(([id,lab])=>(
          <button key={id} onClick={()=>setView(id)} style={{fontSize:13,padding:"9px 18px",borderRadius:9,cursor:"pointer",fontWeight:600,
            border:`1px solid ${view===id?MOSS:LINE_GRID}`,background:view===id?"rgba(225,0,15,0.07)":"transparent",color:view===id?MOSS:SAGE}}>{lab}</button>))}</div>

      {view==="stock"&&(
        <div style={{display:"flex",alignItems:"center",gap:16,flexWrap:"wrap",background:CARD,border:`1px solid ${v.color}55`,borderRadius:14,padding:"15px 20px",marginBottom:16}}>
          <div style={{width:12,height:12,borderRadius:99,background:v.color,boxShadow:`0 0 14px ${v.color}`}}/>
          <div style={{flex:1,minWidth:200}}><div style={{fontSize:12,color:TEXT_MUTE}}>Ziel 2030 · {fmt(p.targetCount)} Bäume</div>
            <div style={{fontFamily:"var(--display)",fontSize:23,fontWeight:600,color:v.color}}>{v.label}</div>
            <div style={{fontSize:12.5,color:TEXT_MUTE}}>{v.note}</div></div>
          <div style={{textAlign:"right",fontFamily:"var(--mono)"}}><div style={{fontSize:12,color:TEXT_MUTE}}>Median 2030</div>
            <div style={{fontSize:25,fontWeight:600,color:PAPER,fontVariantNumeric:"tabular-nums"}}>{fmt(t2030.p50)}</div></div></div>)}

      <div className="dash-grid" style={{display:"grid",gridTemplateColumns:"1fr",gap:18}}>
        <div>
          {view==="stock"?(<>
            <div style={{background:PANEL,border:`1px solid ${LINE_GRID}`,borderRadius:16,padding:"18px 14px 8px"}}>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"baseline",padding:"0 8px 8px"}}>
                <div style={{fontSize:13,color:SAGE,fontWeight:600}}>Bestandsentwicklung 2026 – 2126</div>
                <div style={{fontSize:11,color:TEXT_MUTE,fontFamily:"var(--mono)"}}>Median · 50 % · 90 % Band</div></div>
              <ResponsiveContainer width="100%" height={400}>
                <ComposedChart data={series} margin={{top:8,right:14,bottom:4,left:6}}>
                  <CartesianGrid stroke={LINE_GRID} strokeDasharray="2 4" vertical={false}/>
                  <XAxis dataKey="year" tick={{fill:TEXT_MUTE,fontSize:11,fontFamily:"var(--mono)"}} ticks={[2026,2051,2076,2101,2126]} stroke={LINE_GRID}/>
                  <YAxis domain={[0,yMax]} tick={{fill:TEXT_MUTE,fontSize:11,fontFamily:"var(--mono)"}} tickFormatter={x=>(x/1000).toFixed(0)+"k"} stroke={LINE_GRID} width={38}/>
                  <Tooltip content={<ChartTip target={p.targetCount}/>}/>
                  <Area dataKey="band90" stroke="none" fill={MOSS} fillOpacity={0.1} isAnimationActive={false}/>
                  <Area dataKey="band50" stroke="none" fill={MOSS} fillOpacity={0.22} isAnimationActive={false}/>
                  <Line dataKey="p50" stroke={MOSS} strokeWidth={2.4} dot={false} isAnimationActive={false}/>
                  <ReferenceLine y={p.targetCount} stroke={PAPER} strokeDasharray="5 4" strokeWidth={1.3} label={{value:`Ziel ${fmt(p.targetCount)}`,fill:PAPER,fontSize:11,position:"insideTopRight",fontFamily:"var(--mono)"}}/>
                  {MILESTONES.map(o=><ReferenceLine key={o} x={yearAt(o)} stroke={LINE_GRID} strokeDasharray="1 5"/>)}
                  <ReferenceDot x={2030} y={t2030.p50} r={4} fill={v.color} stroke="#fff" strokeWidth={2} isAnimationActive={false}/>
                </ComposedChart></ResponsiveContainer></div>
            <div className="flex" style={{gap:12,marginTop:14,flexWrap:"wrap"}}>{ms.map(({off,pt})=><MilestoneCard key={off} off={off} pt={pt} target={p.targetCount}/>)}</div>
            <div style={{marginTop:14,background:CARD,border:`1px solid ${LINE_GRID}`,borderRadius:12,padding:"14px 18px"}}>
              <div className="flex" style={{justifyContent:"space-between",alignItems:"flex-end",flexWrap:"wrap",gap:14}}>
                <div style={{flex:"1 1 240px"}}>
                  <div style={{fontSize:11,color:TEXT_MUTE,letterSpacing:1,textTransform:"uppercase",marginBottom:4}}>Kalibrierte Alters-Hazard (0–150 J)</div>
                  <HazardSpark/>
                  <div style={{fontSize:10.5,color:TEXT_MUTE,fontFamily:"var(--mono)",marginTop:2}}>jung ~0.3 % · reif ~1.5 % · Skala 0–4 %/J</div></div>
                <div style={{fontFamily:"var(--mono)",textAlign:"right"}}>
                  <div style={{fontSize:11,color:TEXT_MUTE}}>Ø Ausfall heute · {fmt(deathsY1)} Bäume/J</div>
                  <div><span style={{color:MOSS,fontSize:18,fontWeight:600}}>{(hToday*100).toFixed(2)} %</span>
                    <span style={{color:TEXT_MUTE}}> → </span><span style={{color:AMBER,fontSize:18,fontWeight:600}}>{(hEnd*100).toFixed(2)} %</span></div>
                  <div style={{fontSize:11,color:TEXT_MUTE,marginTop:6}}>Endbestand Median {fmt(endPt.p50)}</div></div></div></div>
          </>):(<>
            <div style={{background:PANEL,border:`1px solid ${LINE_GRID}`,borderRadius:16,padding:"18px 20px"}}>
              <div className="flex items-baseline" style={{justifyContent:"space-between",marginBottom:14,flexWrap:"wrap",gap:10}}>
                <div><div style={{fontSize:13,color:SAGE,fontWeight:600}}>Altersstruktur</div>
                  <div style={{fontSize:11,color:TEXT_MUTE,fontFamily:"var(--mono)",marginTop:2}}><span style={{color:BARK}}>■</span> Bestand 2026 · <span style={{color:MOSS}}>■</span> seit 2026 gepflanzt</div></div>
                <div style={{textAlign:"right"}}><div style={{fontSize:11,color:TEXT_MUTE}}>angezeigtes Jahr</div>
                  <div style={{fontFamily:"var(--display)",fontSize:34,fontWeight:600,color:PAPER,lineHeight:1}}>{yearAt(yOff)}</div></div></div>
              <input type="range" min={0} max={YEARS} step={1} value={yOff} onChange={e=>set("viewYearOff")(parseInt(e.target.value))} style={{width:"100%",accentColor:MOSS,height:4,cursor:"pointer",marginBottom:18}}/>
              <Pyramid classOld={det.classOld[yOff]} classNew={det.classNew[yOff]}/></div>
            <div className="flex" style={{gap:12,marginTop:14,flexWrap:"wrap"}}>
              {[["Gesamtbestand",fmt(det.total[yOff]),PAPER],["Medianalter",det.medAge[yOff]+" J",MOSS],
                ["Anteil < 20 J",Math.round(det.young[yOff]*100)+" %",SAGE],["Anteil ≥ 60 J",Math.round(det.old60[yOff]*100)+" %",AMBER],
                ["seit 2026 gepflanzt",Math.round(renew*100)+" %",MOSS]].map(([l,val,c])=>(
                <div key={l} style={{background:CARD,border:`1px solid ${LINE_GRID}`,borderRadius:12,padding:"12px 15px",flex:"1 1 0",minWidth:110}}>
                  <div style={{fontSize:11,color:TEXT_MUTE}}>{l}</div>
                  <div style={{fontFamily:"var(--display)",fontSize:24,fontWeight:600,color:c,fontVariantNumeric:"tabular-nums"}}>{val}</div></div>))}</div>
            <div style={{marginTop:14,background:PANEL,border:`1px solid ${LINE_GRID}`,borderRadius:16,padding:"16px 14px 8px"}}>
              <div style={{fontSize:13,color:SAGE,fontWeight:600,padding:"0 8px 8px"}}>Medianalter & Altersanteile über die Zeit</div>
              <ResponsiveContainer width="100%" height={230}>
                <ComposedChart data={ageLine} margin={{top:6,right:14,bottom:4,left:6}}>
                  <CartesianGrid stroke={LINE_GRID} strokeDasharray="2 4" vertical={false}/>
                  <XAxis dataKey="year" tick={{fill:TEXT_MUTE,fontSize:11,fontFamily:"var(--mono)"}} ticks={[2026,2051,2076,2101,2126]} stroke={LINE_GRID}/>
                  <YAxis yAxisId="l" tick={{fill:TEXT_MUTE,fontSize:11,fontFamily:"var(--mono)"}} stroke={LINE_GRID} width={32}/>
                  <YAxis yAxisId="r" orientation="right" domain={[0,100]} tick={{fill:TEXT_MUTE,fontSize:11,fontFamily:"var(--mono)"}} tickFormatter={x=>x+"%"} stroke={LINE_GRID} width={40}/>
                  <Tooltip contentStyle={{background:"#ffffff",border:`1px solid ${LINE_GRID}`,borderRadius:9,fontFamily:"var(--mono)",fontSize:12}} labelStyle={{color:MOSS}}/>
                  <Line yAxisId="l" dataKey="med" name="Medianalter (J)" stroke={MOSS} strokeWidth={2.2} dot={false} isAnimationActive={false}/>
                  <Line yAxisId="r" dataKey="young" name="< 20 J (%)" stroke={GREEN} strokeWidth={1.6} strokeDasharray="4 3" dot={false} isAnimationActive={false}/>
                  <Line yAxisId="r" dataKey="old" name="≥ 60 J (%)" stroke={AMBER} strokeWidth={1.6} strokeDasharray="4 3" dot={false} isAnimationActive={false}/>
                  <ReferenceLine x={yearAt(yOff)} yAxisId="l" stroke={PAPER} strokeOpacity={.4}/>
                </ComposedChart></ResponsiveContainer></div>
          </>)}
        </div>

        <div style={{background:PANEL,border:`1px solid ${LINE_GRID}`,borderRadius:16,padding:"20px 20px 8px"}}>
          <div className="flex" style={{gap:7,flexWrap:"wrap",marginBottom:18}}>
            {Object.keys(PRESETS).map(n=>(<button key={n} onClick={()=>preset(n)} style={{fontSize:12,padding:"7px 12px",borderRadius:8,cursor:"pointer",border:`1px solid ${MOSS}40`,background:"rgba(225,0,15,0.05)",color:MOSS,fontWeight:600,textTransform:"capitalize"}}>{n}</button>))}
            <button onClick={()=>setP({...DEFAULTS,viewYearOff:p.viewYearOff})} style={{fontSize:12,padding:"7px 12px",borderRadius:8,cursor:"pointer",border:`1px solid ${LINE_GRID}`,background:"transparent",color:SAGE}}>Zurücksetzen</button></div>
          <Section title="Bestand & Ziel">
            <Slider label="Startbestand 2026" value={p.N0} min={12000} max={20000} step={100} onChange={set("N0")} display={fmt(p.N0)} hint="Kataster: 16'621 aktiv"/>
            <Slider label="Zielbestand 2030" value={p.targetCount} min={14000} max={20000} step={250} onChange={set("targetCount")} display={fmt(p.targetCount)}/>
          </Section>
          <Section title="Ausfall & Klima">
            <div style={{fontSize:11.5,color:TEXT_MUTE,marginBottom:12,lineHeight:1.5}}>Basis-Ausfall = kalibrierte Alters-Hazard (fix). Klima = TreeGOER-Exceedance gegen Winterthurs Zukunfts-bio05.</div>
            <Segmented label="Klimaszenario (CitiesGOER)" value={p.scenario}
              options={[{value:"ssp126",label:"SSP1-2.6"},{value:"ssp370",label:"SSP3-7.0"},{value:"ssp585",label:"SSP5-8.5"}]} onChange={set("scenario")}/>
            <Segmented label="TreeGOER-Quantil (tree_q)" value={p.treeQ}
              options={[{value:"q95",label:"q95 · milder"},{value:"qrt3",label:"qrt3 · strenger"}]} onChange={set("treeQ")}/>
            <Segmented label="TreeGOER-Stärke (tree_k)" value={p.treeK}
              options={[{value:0.08,label:"0.08"},{value:0.15,label:"0.15"}]} onChange={v=>set("treeK")(parseFloat(v))}/>
            <div style={{fontSize:11,color:SAGE,fontFamily:"var(--mono)",margin:"-4px 0 14px"}}>climate_mult = ×{climateMult.toFixed(3)} ({SCEN_LABEL[p.scenario]}, 45 % Artabdeckung)</div>
            <Slider label="Zusätzlicher Standort-/Stressfaktor" value={p.manualStress} min={0.7} max={2.0} step={0.05} onChange={set("manualStress")} display={"×"+p.manualStress.toFixed(2)} hint="für nicht erfasste lokale Stressoren; ×1.6 ≈ ~210 Fäll./J heute"/>
            <Slider label="Klima-Trend bis 2126" value={p.climateTrendEnd} min={0} max={1} step={0.05} onChange={set("climateTrendEnd")} display={"+"+(p.climateTrendEnd*100).toFixed(0)+" %"} hint="wirkt als ×(1+trend), linear über Zeit"/>
            <Slider label="Standortfaktor (^1.0)" value={p.siteFactor} min={0.7} max={1.5} step={0.05} onChange={set("siteFactor")} display={"×"+p.siteFactor.toFixed(2)}/>
            <Slider label="Managementfaktor (^0.6)" value={p.mgmtFactor} min={0.7} max={1.3} step={0.05} onChange={set("mgmtFactor")} display={"×"+p.mgmtFactor.toFixed(2)} hint="<1 = bessere Pflege"/>
          </Section>
          <Section title="Ersatz & Neupflanzung">
            <Slider label="Ersatzrate" value={p.replacementRate} min={0} max={1} step={0.05} onChange={set("replacementRate")} display={(p.replacementRate*100).toFixed(0)+" %"}/>
            <Slider label="Ersatzverzögerung" value={p.replacementDelay} min={0} max={5} step={1} onChange={set("replacementDelay")} display={p.replacementDelay+" J"}/>
            <Slider label="Zusatzpflanzungen / Jahr" value={p.annualNewTrees} min={0} max={800} step={25} onChange={set("annualNewTrees")} display={fmt(p.annualNewTrees)}/>
            <Slider label="Pflanzfenster von" value={p.newStart} min={1} max={10} step={1} onChange={set("newStart")} display={yearAt(p.newStart).toString()}/>
            <Slider label="Pflanzfenster bis" value={p.newEnd} min={1} max={30} step={1} onChange={set("newEnd")} display={yearAt(p.newEnd).toString()}/>
            <Slider label="Pflanzalter neue Bäume" value={p.initAge} min={1} max={20} step={1} onChange={set("initAge")} display={p.initAge+" J"} hint="new_tree_initial_age (Default 10)"/>
            <Segmented label="Pflanzstrategie" value={p.strategy} options={Object.keys(STRAT_NEW).map(k=>({value:k,label:STRAT_LABEL[k]}))} onChange={set("strategy")}/>
          </Section>
          <Section title="Lebensdauer & Simulation">
            <div style={{fontSize:11.5,color:TEXT_MUTE,marginBottom:12,lineHeight:1.5}}>life_mult = (130/Lebensdauer)^Gewicht, gekappt 0.5–2.0. Bestandsmittel ≈ 1.0 (Hauptwirkung: Unsicherheit).</div>
            <Slider label="Lebensdauer-Gewicht" value={p.lifeHazardWeight} min={0} max={1} step={0.1} onChange={set("lifeHazardWeight")} display={p.lifeHazardWeight.toFixed(1)}/>
            <Segmented label="Unsicherheitsmodus" value={p.lifeMode} options={[{value:"none",label:"none"},{value:"per_run",label:"per_run"}]} onChange={set("lifeMode")}/>
            <Slider label="Monte-Carlo-Läufe" value={p.nRuns} min={50} max={400} step={50} onChange={set("nRuns")} display={p.nRuns.toString()}/>
            <button onClick={()=>set("seed")(Math.floor(Math.random()*1e9))} style={{fontSize:12,padding:"7px 14px",borderRadius:8,cursor:"pointer",width:"100%",border:`1px solid ${LINE_GRID}`,background:"transparent",color:SAGE,marginTop:2}}>Zufall neu würfeln</button>
          </Section>
        </div>
      </div>

      <p style={{color:TEXT_MUTE,fontSize:12,lineHeight:1.6,marginTop:22,maxWidth:920}}>
        Repliziert die Engine aus winterthur_tree_stochastic_goal_planning_v7.py: p_fail = clip(base_p(Alter) · climate_mult · (1+trend) · Standort^1 · Management^0.6 · life_mult, 0.0001, 0.50).
        base_p = kalibrierte Alters-Hazard aus dem Kataster (Sterbetafel, Laplace α=0.5, min_risk_set=20, clip ≤ 0.35). climate_mult = bestandsgewichtete TreeGOER-Exceedance:
        exp(−tree_k·max(0, Winterthurs Zukunfts-bio05 − bio05-Grenze der Art)), 1/clip(.,0.2,2.0) — aus CitiesGOER (SSP126: 26.9 °C) und TreeGOER (45 % Artabdeckung, Rest neutral).
        Unter SSP126 ist climate_mult ≈ 1.0, der Klimaeffekt kommt also fast ganz aus dem climate_trend-Ramp. Neue Bäume starten mit Alter {p.initAge}, Ersatz = round(Ausfälle·Rate) nach Verzögerung.
        Bänder aus {p.nRuns} Läufen. Noch aggregiert statt artweise: base_p ist global (nicht je Art) und der CityTrees-Proxy wird durch TreeGOER überschrieben — wie in deiner Config.
      </p>
    </div>
    <style>{`@media(min-width:1000px){.dash-grid{grid-template-columns:1.55fr 1fr!important;}}`}</style>
  </div>);
}
