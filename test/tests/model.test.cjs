/* Ohne Abhängigkeiten ausführen: node test/tests/model.test.cjs */
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const base = join(__dirname,'..');
const source = ['data.js','model.js'].map(f=>readFileSync(join(base,f),'utf8')).join('\n');
const ctx=vm.createContext({console});
vm.runInContext(source,ctx);
const value = expression => vm.runInContext(expression,ctx);
const within=(x,y,tol=1e-6)=>assert.ok(Math.abs(x-y)<tol,`${x} != ${y}`);
const scenarios = [
  "simulate({...baseParams},'neutral',observedMix())",
  "simulate({...baseParams,year:2050,plantings:110,start:2027,end:2050,replace:0.75,delay:2,est:0.8},'neutral',observedMix())",
  "simulate({...baseParams,year:2050,plantings:110,start:2027,end:2050,replace:0.75,delay:2,est:0.8,scenario:'ssp585'},'proxy',observedMix())",
];
assert.equal(value('START'),16621);
assert.equal(value('sum(A)'),16620);
assert.equal(value('sum(species.map(z=>z.count))'),16621);
assert.equal(value('runModelChecks().every(z=>z.pass)'),true);

for(const sc of scenarios){
 const data=value(sc);
 assert.equal(data.length,101);
 within(data[0].total,16621);
 for(let i=1;i<data.length;i++){
   let z=data[i],prev=data[i-1];
   assert.ok(z.total>=0 && z.deaths>=0);
   within(z.total,z.hist.reduce((a,b)=>a+b,0));
   within(z.total,z.composition.reduce((a,b)=>a+b,0));
   const est=sc.includes('est:0.8')?0.8:1;
   within(z.total,prev.total-z.deaths+(z.extra+z.replacement)*est);
 }
}
// Ändert man im Neutralmodus nur die Arten der Pflanzungen, ändert sich die Gesamtzahl nicht.
const neutralA=value("simulate({...baseParams},'neutral',observedMix())");
const neutralB=value("simulate({...baseParams},'neutral',[{name:'Quercus robur',weight:1}])");
for(let i=0;i<neutralA.length;i++) within(neutralA[i].total,neutralB[i].total);
const until2030=value("simulate({...baseParams},'neutral',observedMix(),2030)");
assert.equal(until2030.length,5);
within(until2030[4].total,neutralA[4].total);
console.log('✓ 3 Szenarien mit 101 Jahren: Bestand/Alter/Arten und Jahresbilanz konsistent');
console.log('✓ Artenneutralität, Anfangsbestand und Zieljahr-Abkürzung korrekt');
console.log('✓ Tests erfolgreich');
