/** Editable paper evidence. Every measured point is read from retained result JSON. */
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { execFileSync } from 'node:child_process';

const here = path.dirname(fileURLToPath(import.meta.url));
const workspaceDir = path.dirname(here);
const build = path.join(workspaceDir, '.build_www', 'paper_evidence');
const runtime = 'C:/Users/hejie/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const skill = 'C:/Users/hejie/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations';
const python = runtime + '/python/python.exe';
process.env.NODE_PATH = runtime + '/node/node_modules';
process.env.RUNTIME_NODE_MODULES = runtime + '/node/node_modules';
process.env.RUNTIME_NODE = runtime + '/node/bin/node.exe';
process.env.RUNTIME_PYTHON = python;
const { Presentation, PresentationFile } = await import(pathToFileURL(runtime + '/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs').href);
const { applyPresentationChartFont, finalizePresentation } = await import(pathToFileURL(skill + '/container_tools/artifact_tool_utils.mjs').href);
await fs.mkdir(build, { recursive: true });
const presentation = Presentation.create({ slideSize: { width: 1408, height: 600 } });
const font = 'Arial';
const colors = { coral: '#DF715D', blue: '#487EB5', mint: '#56A68C', purple: '#8C77B4', gold: '#CBA24C', ink: '#25344B', gray: '#8C95A4', grid: '#DEE4ED' };
const textStyle = { typeface: font, fontSize: 24, fill: colors.ink };
function text(slide, value, x, y, w, h, size=26, bold=false, align='center', color=colors.ink) {
  const shape=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  shape.text=value; shape.text.style={typeface:font,fontSize:size,bold,color,alignment:align,verticalAlignment:'middle',autoFit:'none'}; return shape;
}
function title(slide,value){}
function line(slide,x1,y1,x2,y2,color=colors.ink,width=1.5,dashed=false){
  return slide.shapes.add({geometry:'line',position:{left:Math.min(x1,x2),top:Math.min(y1,y2),width:Math.abs(x2-x1)||0.001,height:Math.abs(y2-y1)||0.001,verticalFlip:y2<y1},fill:'none',line:{fill:color,width,style:dashed?'dashed':'solid'}});
}
function axis(titleValue,limits={}){return {visible:true,...(titleValue?{title:{text:titleValue,textStyle:{...textStyle,fontSize:25}}}:{}),textStyle:{...textStyle,fontSize:21},line:{fill:colors.ink,width:1.4},majorGridlines:{fill:colors.grid,width:0.8},...limits};}
function chart(slide,type,config){
  // Current artifact-tool requires explicit dash style on point outlines.
  for(const s of config.series??[])for(const p of s.points??[])if(p.line&&!p.line.style)p.line.style='solid';
  // Excel stores 15 significant digits. Preserve all source points at that
  // precision so PowerPoint's embedded workbook and chart cache stay identical.
  for(const s of config.series??[])for(const key of ['values','xValues'])if(s[key])s[key]=s[key].map(v=>Number(v.toPrecision(15)));
  const c=slide.charts.add(type,{chartFill:'#FFFFFF',chartLine:{fill:'none',width:0},plotAreaFill:'#FFFFFF',plotAreaLine:{fill:'none',width:0},hasLegend:false,...config});applyPresentationChartFont(c,{fontFamily:font});return c;
}
const mean=a=>a.reduce((x,y)=>x+y,0)/a.length;
const readJson=async f=>JSON.parse((await fs.readFile(f,'utf8')).replace(/^\uFEFF/,''));
const out=path.join(here,'outputs','www_v1','v7');
const current=await readJson(path.join(here,'paper_results_20261004.json'));
const evidence={cross:{},ablation:{},buckets:{},sensitivity:{},robustness:{},native:{}};

// 1: Conceptual motivation, with the same 200 samples and formulas as the source.
{
 const slide=presentation.slides.add(); slide.background.fill='#FFFFFF';
 const freq=Array.from({length:200},(_,i)=>10**(3.3*i/199));
 const q=freq.map(f=>1/Math.sqrt(f+1)), qmin=Math.min(...q), qmax=Math.max(...q);
 const aware=q.map(v=>0.8*(1-0.75*((v-qmin)/(qmax-qmin))**0.7));
 const eta=Array.from({length:200},(_,i)=>0.2+2*i/199), rho=eta.map(v=>0.2*Math.max(0,Math.min(1,(1.5-v)/0.5)));
 title(slide,'Evidence-aware adaptation');
 chart(slide,'scatter',{position:{left:16,top:8,width:678,height:575},title:'(a) Item-level routing',titleTextStyle:{...textStyle,fontSize:28,bold:true},scatterOptions:{style:'line'},
  series:[{name:'Uniform adaptation',xValues:freq,values:freq.map(()=>0.8),line:{fill:colors.gray,width:3,style:'dashed'},marker:{symbol:'none'}},{name:'FedPASA (evidence-aware)',xValues:freq,values:aware,line:{fill:colors.coral,width:4},marker:{symbol:'none'}}],
  xAxis:axis('Item interaction frequency (log scale)',{min:1,max:2200,numberFormatCode:'0',tickLabelPosition:'low'}),yAxis:axis('Behavioral adaptation strength',{min:0,max:0.9,majorUnit:0.2,numberFormatCode:'0.0'}),hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{...textStyle,fontSize:22}}});
 chart(slide,'scatter',{position:{left:714,top:8,width:678,height:575},title:'(b) Density-gated anchoring',titleTextStyle:{...textStyle,fontSize:28,bold:true},scatterOptions:{style:'line'},
  series:[{name:'No anchoring',xValues:eta,values:eta.map(()=>0),line:{fill:colors.gray,width:3,style:'dashed'},marker:{symbol:'none'}},{name:'AutoAnchor ρeff',xValues:eta,values:rho,line:{fill:colors.mint,width:4},marker:{symbol:'none'}}],
  xAxis:axis('Target density ηT = |UT| / |IT|',{min:0,max:2.2,majorUnit:0.5,numberFormatCode:'0.0',tickLabelPosition:'low'}),yAxis:axis('Anchor weight',{min:0,max:0.22,majorUnit:0.05,numberFormatCode:'0.00'}),hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{...textStyle,fontSize:22}}});
 slide.speakerNotes.textFrame.setText('Conceptual functions from www_workspace/make_motivation_figure.py. These are illustrative mechanism curves, not measured experimental results. 200 original sampling points retained per curve.');
}

// 2: Source reliability, retaining the six existing values exactly.
{
 const data=await readJson(path.join(here,'outputs','www_v1','source_gate.json'));
 const slide=presentation.slides.add();slide.background.fill='#FFFFFF'; title(slide,'Source reliability gate');
 const pos={left:90,top:12,width:1228,height:575};
 const series=data.points.map(([name,x,y],i)=>({name:name.replace('_to_',' → '),xValues:[x],values:[y*100],fill:y>0?colors.mint:colors.coral,line:{fill:'none',width:0},marker:{symbol:'circle',size:12},dataLabelOverrides:[{idx:0,text:name.replace('_to_',' → '),position:i===4?'b':'t',showValue:false,showSeriesName:true,textStyle:{...textStyle,fontSize:21}}]}));
 series.push({name:'Zero source benefit',xValues:[0.191,0.215],values:[0,0],line:{fill:colors.gray,width:1.5},marker:{symbol:'none'}});
 series.push({name:'τ = 0.2025',xValues:[data.tau,data.tau],values:[-6.5,2],line:{fill:colors.blue,width:2.4,style:'dashed'},marker:{symbol:'none'}});
 chart(slide,'scatter',{position:pos,scatterOptions:{style:'lineWithMarkers'},series,
 xAxis:axis('Source adapter zero-shot HR@20 on target validation',{min:0.190,max:0.215,majorUnit:0.005,numberFormatCode:'0.000',tickLabelPosition:'low'}),
 yAxis:axis('Source benefit (src − no-src), ×10⁻²',{min:-6,max:2,majorUnit:2,numberFormatCode:'0.0'}),dataLabels:{showValue:false,showSeriesName:false,textStyle:{...textStyle,fontSize:21}}});
 text(slide,'Exploratory threshold\nτ = 0.2025 (post hoc)',800,425,390,80,25,false,'left',colors.blue);
 slide.speakerNotes.textFrame.setText('Exploratory source diagnostic from www_workspace/outputs/www_v1/source_gate.json. x is recorded seed-42 zero-shot target-validation HR@20. y is recorded five-seed mean source minus no-source HR@20, displayed ×100. Original six values preserved. Threshold 0.2025 is post hoc; no independent threshold-selection record is available.');
}

// 3: Common nonempty runs; omitted hit keys mean zero hits, not missing trials.
{
 const methods=['transfr','fedmoe','fedpasa'],buckets=['zero','1_5','6_20','20plus'];
 const names=['music_to_books','music_to_movies','books_to_movies','books_to_music','movies_to_books','movies_to_music'],rows=[];
 for(const direction of names)for(const seed of [42,123,456,789,2024]){const row=[];for(const method of methods)row.push((await readJson(path.join(out,'cross',direction,method,`seed${seed}`,'results.json'))).metrics_no_pap);rows.push(row);}
 const data={hr:[],ndcg:[]},supports=[];
 for(const bucket of buckets){const selected=rows.filter(row=>row.every(m=>m[`freq_${bucket}_count`]>0));if(!selected.length)throw new Error('No shared nonempty run: '+bucket);supports.push(selected.length);
  for(const [key,metric] of [['hr','HR@20'],['ndcg','NDCG@20']])data[key].push(methods.map((_,i)=>mean(selected.map(row=>row[i][`s99freq_${bucket}_${metric}`]??0))));
 }
 if(supports.join(',')!=='20,30,30,30')throw new Error('Unexpected bucket support');
 evidence.buckets={methods,buckets,metrics:['HR@20','NDCG@20'],data,supports,seeds:[42,123,456,789,2024],protocol:'sampled-99; non-personalized base; common nonempty runs'};
 const slide=presentation.slides.add();slide.background.fill='#FFFFFF';
 for(let metric=0;metric<2;metric++){const x=metric?712:0,values=metric?data.ndcg:data.hr;text(slide,metric?'(b) NDCG@20':'(a) HR@20',x,0,696,40,28,true);
  chart(slide,'bar',{position:{left:x,top:42,width:696,height:552},categories:['0','1–5','6–20','>20'],series:methods.map((m,i)=>({name:['TransFR','FedMoE','FedPASA (base)'][i],values:values.map(v=>v[i]),fill:[colors.purple,colors.mint,colors.blue][i],line:{fill:'none',width:0}})),barOptions:{direction:'column',grouping:'clustered',gapWidth:65},
   xAxis:axis('Target item frequency',{majorGridlines:null,textStyle:{...textStyle,fontSize:24}}),yAxis:axis(metric?'NDCG@20':'HR@20',{min:0,max:metric?0.45:0.7,majorUnit:0.1,numberFormatCode:'0.0'}),hasLegend:true,legend:{position:'top',overlay:false,textStyle:{...textStyle,fontSize:23}}});
 }
 slide.speakerNotes.textFrame.setText('Corrected macro means of common nonempty runs in outputs/www_v1/cross/{direction}/{transfr,fedmoe,fedpasa}/seed{42,123,456,789,2024}/results.json, metrics_no_pap. FedPASA is the unpersonalized base. Require freq_{bucket}_count>0 for all methods; missing s99freq hit keys are zero. Common run counts:20,30,30,30. Unseen mean support in its20 nonempty runs is13 test cases. All outcomes retained including the mid-frequency bucket where base is lower.');
}

// 4: Real sensitivity runs plus the protocol-identical baseline operating point.
{
 const slide=presentation.slides.add();slide.background.fill='#FFFFFF';title(slide,'Sensitivity on Music → Movies (three seeds)');
 const seeds=[42,123,456],defaults=[0.85,0.5,0.2,0.3];
 const baseline=await Promise.all(seeds.map(seed=>readJson(path.join(out,'cross','music_to_movies','fedpasa',`seed${seed}`,'results.json'))));
 const paramFields=['fedbat_alpha','fedbat_cold_power','fedbat_condition_scale','semantic_anchor_weight'];
 const sweeps=[['alpha','α (residual mixture)',[0.55,0.7,0.85,1],colors.coral],['coldpower','κ (coldness power)',[0.35,0.5,0.7,1],colors.purple],['condscale','λ (conditioning)',[0,0.1,0.2,0.4],colors.blue],['anchorw','ρ (max anchor)',[0,0.1,0.2,0.3,0.5],colors.mint]];
 for(let i=0;i<sweeps.length;i++){const [name,label,values,color]=sweeps[i],xs=[],ys=[],ndcgs=[];
  for(const val of values){const spelling=Number.isInteger(val)?val.toFixed(1):String(val),pts=[],ptsN=[];
   for(let j=0;j<seeds.length;j++){const r=val===defaults[i]?baseline[j]:await readJson(path.join(out,'sensitivity',name+spelling,`seed${seeds[j]}`,'results.json'));
    const ignored=new Set(['output_dir',paramFields[i],...(i===3?['effective_semantic_anchor_weight']:[])]);
    for(const key of new Set([...Object.keys(r.args),...Object.keys(baseline[j].args)]))if(!ignored.has(key)&&JSON.stringify(r.args[key])!==JSON.stringify(baseline[j].args[key]))throw new Error(`Sensitivity protocol mismatch: ${name}/${seeds[j]}/${key}`);
    const y=r.metrics_no_pap['s99_HR@20'];if(!Number.isFinite(y))throw new Error('Missing true HR@20');pts.push(y);ptsN.push(r.metrics_no_pap['s99_NDCG@20']);
   }xs.push(val);ys.push(mean(pts));ndcgs.push(mean(ptsN));
  }
  evidence.sensitivity[name]={values:xs,HR20:ys,NDCG20:ndcgs,seeds,default:defaults[i],protocol:'sampled-99; Music to Movies; non-personalized base'};
  const defaultIndex=xs.indexOf(defaults[i]);
  chart(slide,'scatter',{position:{left:12+702*(i%2),top:6+296*Math.floor(i/2),width:680,height:292},scatterOptions:{style:'lineWithMarkers'},series:[{name:'FedPASA (base)',xValues:xs,values:ys,fill:color,line:{fill:color,width:3.5},marker:{symbol:'circle',size:9},points:[{idx:defaultIndex,fill:'#FFFFFF',line:{fill:color,width:2}}],dataLabelOverrides:[{idx:defaultIndex,text:'default',showValue:false,position:'t',textStyle:{...textStyle,fontSize:18}}]}],dataLabels:{showValue:false},
  xAxis:axis(label,{min:Math.min(...xs),max:Math.max(...xs),numberFormatCode:'0.00',tickLabelPosition:'low',textStyle:{...textStyle,fontSize:18},title:{text:label,textStyle:{...textStyle,fontSize:23}}}),yAxis:axis('HR@20',{min:0.4,max:0.52,majorUnit:0.02,numberFormatCode:'0.00',textStyle:{...textStyle,fontSize:19}})});
 }
 slide.speakerNotes.textFrame.setText('Source: completed outputs/www_v1/v7 records. Five seeds for the main matrix and three for sensitivity/client studies. Selected alpha=.85, kappa=.5, lambda=.2, rho=.3. Target Books uses 8 rounds x30 local steps; other cross-domain targets use 12x60. All points are measured; the hollow sensitivity marker is the selected default.');
}

// The main-result table reports the personalized deployed system. Component
// controls below instead share the non-personalized base, avoiding confounding.
const directions=['music_to_books','music_to_movies','books_to_movies','books_to_music','movies_to_books','movies_to_music'];
const directionLabels=['Music\n→ Books','Music\n→ Movies','Books\n→ Movies','Books\n→ Music','Movies\n→ Books','Movies\n→ Music'];
const fiveSeeds=[42,123,456,789,2024];
const mainMethods=['fcf','scratch','fedprox','fedadam','transfr','l2sp','fedper','pfedme','fedmoe','fedpasa'];
const methodLabels=['FedMF (FCF)','FedAvg-Adapter','FedProx','FedAdam','TransFR control','L2-SP','FedPer','pFedMe-style','FedMoE','FedPASA'];
async function runMetric(method,direction,seed,personalized=false){
 const relative=personalized?`ditto/${direction}/pasa_ditto/seed${seed}/results.json`:`cross/${direction}/${method}/seed${seed}/results.json`;
 const r=await readJson(path.join(out,relative));
 if(r.args.seed!==seed||r.args.eval_sampled_negatives!==99||r.args.eval_users!==5000)throw new Error('Protocol mismatch: '+relative);
 const expected=direction.endsWith('_books')?[8,30]:[12,60];
 if(r.args.rounds!==expected[0]||r.args.local_steps!==expected[1])throw new Error('Budget mismatch: '+relative);
 const key=personalized||method==='pfedme'?'metrics_ditto':method==='fcf'?'metrics_fedmf':'metrics_no_pap';
 const m=r[key];if(!m||!Number.isFinite(m['s99_HR@20'])||!Number.isFinite(m['s99_NDCG@20']))throw new Error('Missing measured metric: '+relative);
 return m;
}
function mixColor(a,b,t){const aa=a.slice(1).match(/../g).map(v=>parseInt(v,16)),bb=b.slice(1).match(/../g).map(v=>parseInt(v,16));return '#'+aa.map((v,i)=>Math.round(v+(bb[i]-v)*t).toString(16).padStart(2,'0')).join('');}
// 5: Full ten-method, six-direction results. Native monochrome three-rule tables.
{
 const slide=presentation.slides.add();slide.background.fill='#FFFFFF';
 const cube=[];
 for(const method of mainMethods){const row=[];for(const d of directions){const runs=await Promise.all(fiveSeeds.map(seed=>runMetric(method,d,seed,method==='fedpasa')));row.push(['s99_HR@20','s99_NDCG@20'].map(k=>mean(runs.map(r=>r[k]))));}cube.push(row);}
 evidence.cross={methods:methodLabels,directions,metrics:['HR@20','NDCG@20'],values:cube,seeds:fiveSeeds,protocol:'sampled-99; FedPASA includes final personalization'};
 for(let metric=0;metric<2;metric++){
  const x=metric?718:10;const matrix=cube.map(row=>row.map(cell=>cell[metric]));const min=Math.min(...matrix.flat()),max=Math.max(...matrix.flat());
  text(slide,metric?'(b) NDCG@20':'(a) HR@20',x,1,680,42,28,true);
  const values=[['Method',...directionLabels],...matrix.map((row,i)=>[methodLabels[i],...row.map(v=>v.toFixed(4))])];
  const table=slide.tables.add({rows:11,columns:7,left:x,top:48,width:680,height:493,columnWidths:[188,82,82,82,82,82,82],values});
  table.styleOptions={headerRow:false,bandedRows:false,bandedColumns:false};
  table.borders.assign({fill:'#FFFFFF',width:1.2,style:'solid'});
  table.cells.block({row:0,column:0,rowCount:11,columnCount:7}).assign({textStyle:{typeface:font,fontSize:21,color:colors.ink,alignment:'center'},margins:{left:3,right:3,top:2,bottom:2},anchor:'center'});
  table.rows[0].height=63;
  for(let i=1;i<11;i++)table.rows[i].height=43;
  for(let col=0;col<7;col++){table.getCell(0,col).fill='#FFFFFF';table.getCell(0,col).text.style={typeface:font,fontSize:20,bold:true,color:'#000000',alignment:col?'center':'left'};}
  for(let row=1;row<11;row++){
   const lab=table.getCell(row,0);lab.fill='#FFFFFF';lab.text.style={typeface:font,fontSize:21,bold:row===10,color:'#000000',alignment:'left'};
   for(let col=1;col<7;col++){const v=matrix[row-1][col-1];const cell=table.getCell(row,col);cell.fill='#FFFFFF';cell.text.style={typeface:font,fontSize:22,color:'#000000',bold:v===Math.max(...matrix.map(rr=>rr[col-1])),alignment:'center'};}
  }
  line(slide,x,48,x+680,48,'#000000',1.7);
  line(slide,x,111,x+680,111,'#000000',1.1);
  line(slide,x,498,x+680,498,'#000000',1.1);
  line(slide,x,541,x+680,541,'#000000',1.7);
 }
 slide.speakerNotes.textFrame.setText('Source: completed outputs/www_v1/v7 records. Five seeds for the main matrix and three for sensitivity/client studies. Selected alpha=.85, kappa=.5, lambda=.2, rho=.3. Target Books uses 8 rounds x30 local steps; other cross-domain targets use 12x60. All points are measured; the hollow sensitivity marker is the selected default.');
}
// 6: Component and parameter-matched controls, all on the same base protocol.
{
 const methods=['fedpasa','pasa_no_routing','pasa_no_condition','pasa_no_anchor','pasa_const_routing','pasa_rand_routing','pasa_item_anchor'];
 const labels=['FedPASA (base)','w/o routing','w/o conditioning','w/o AutoAnchor','Constant routing','Random routing','Per-item anchor'];
 const palette=[colors.blue,colors.coral,colors.purple,colors.mint,colors.gold,'#B9A2D3','#73B7C7'];const matrix=[];
 for(const method of methods){const runs=[];for(const d of directions)for(const seed of fiveSeeds)runs.push(await runMetric(method,d,seed));matrix.push(['s99_HR@20','s99_NDCG@20'].map(k=>mean(runs.map(r=>r[k]))));}
 evidence.ablation={methods:labels,metrics:['HR@20','NDCG@20'],values:matrix,seeds:fiveSeeds,directions,protocol:'sampled-99; non-personalized base'};
 const slide=presentation.slides.add();slide.background.fill='#FFFFFF';
 for(let metric=0;metric<2;metric++){
  const x=metric?712:0;text(slide,metric?'(b) NDCG@20':'(a) HR@20',x,0,696,38,28,true);
  chart(slide,'bar',{position:{left:x,top:40,width:696,height:552},categories:labels,series:[{name:'Six-direction mean',values:matrix.map(row=>row[metric]),valuesFormatCode:'0.0000',dataLabelOverrides:matrix.map((row,idx)=>({idx,text:row[metric].toFixed(4),showValue:false,position:'outEnd',textStyle:{...textStyle,fontSize:20}})),fill:colors.blue,line:{fill:'none',width:0},points:palette.map((fill,idx)=>({idx,fill,line:{fill:'none',width:0}}))}],barOptions:{direction:'bar',grouping:'clustered',gapWidth:75},
   xAxis:axis('',{textStyle:{...textStyle,fontSize:22},majorGridlines:null,position:'left'}),yAxis:axis(metric?'NDCG@20':'HR@20',{min:0,max:metric?0.24:0.52,majorUnit:metric?0.04:0.1,numberFormatCode:'0.00',position:'bottom',textStyle:{...textStyle,fontSize:20}}),
   dataLabels:{showValue:true,position:'outEnd',textStyle:{...textStyle,fontSize:20}},hasLegend:false});
 }
 slide.speakerNotes.textFrame.setText('All values from cross/{direction}/{variant}/seed*/results.json metrics_no_pap. Six directions ×five seeds. Base and all six ablations omit the post-federated personalization stage. Includes no-routing, no-conditioning, no-anchor, constant/random parameter-matched routing, and per-item anchor regardless of outcome. Constant/random use same 553280 parameters as base. HR@20 and NDCG@20 are raw rates under sampled-99 ranking, not percentages.');
}
// 7: Selected-setting scalability/non-IID and separately labelled legacy controls.
{
 const slide=presentation.slides.add();slide.background.fill='#FFFFFF';
 const groups=['scalability','noniid','fixedcompute','participation'];
 const titles=['(a) Client scale','(b) Non-IID partitions','(c) Fixed compute (legacy)','(d) Participation (legacy)'];
 const settings=[['clients5','clients20','clients50'],['dirichlet_activity_a0.1','dirichlet_activity_a0.5','dirichlet_popularity_a0.5'],['clients5','clients20','clients50'],['rate0.2','rate0.5','rate1.0']];
 const cats=[['5','20','50'],['Activity .1','Activity .5','Popularity .5'],['5','20','50'],['20%','50%','100%']];
 evidence.robustness=current.studies;
 for(let panel=0;panel<4;panel++){
  const x=panel%2*704,y=Math.floor(panel/2)*300;
  text(slide,titles[panel],x,y,704,30,24,true);
  const rows=settings[panel].map(setting=>setting==='rate1.0'?current.studies.fixedcompute.find(r=>r.setting==='clients5'):current.studies[groups[panel]].find(r=>r.setting===setting));
  const series=['TransFR control','FedPASA (base)'].map((name,i)=>({name,values:rows.map(r=>r.values[i][0]),fill:[colors.purple,colors.blue][i],line:{fill:'none',width:0}}));
  chart(slide,'bar',{position:{left:x+4,top:y+30,width:696,height:268},categories:cats[panel],series,barOptions:{direction:'column',grouping:'clustered',gapWidth:90},
    xAxis:axis(['Clients','Partition','Clients','Participation'][panel],{majorGridlines:null,textStyle:{...textStyle,fontSize:19},title:{text:['Clients','Partition','Clients','Participation'][panel],textStyle:{...textStyle,fontSize:19}}}),
    yAxis:axis('HR@20',{min:0,max:panel<2?.7:.35,majorUnit:panel<2?.2:.1,numberFormatCode:'0.0',textStyle:{...textStyle,fontSize:18},title:{text:'HR@20',textStyle:{...textStyle,fontSize:19}}}),
    hasLegend:true,legend:{position:'top',overlay:false,textStyle:{...textStyle,fontSize:18}}});
 }
 slide.speakerNotes.textFrame.setText('Panels a/b use v7 selected settings, 12x60. Panels c/d use the existing eight-round legacy configuration .55/.7/.4/.2, kept separate. Fixed compute uses 5x30,20x8,50x3 client steps per round. Participation is 20%,50%,100% of five clients; 100% reuses the protocol-identical clients5 control. All are Music to Movies, three seeds, unpersonalized base. Exact values are generated into appendix_settings_right.tex.');
}

// 8: All native full-ranking outputs. Each panel maps one-to-one to the
// nine rows and the four metric columns in tables_native_reproductions.tex.
{
 const nativeLabels={music_to_books:'M–B: B',music_to_movies:'M–V: V',books_to_movies:'B–V: V',books_to_music:'M–B: M',movies_to_books:'B–V: B',movies_to_music:'M–V: M'};
 const rows=current.native.map(r=>({method:r.method,run:nativeLabels[r.direction]??r.direction,values:r.full,seeds:r.seeds}));
 for(const row of rows)if(row.values.some(v=>!Number.isFinite(v)||v<0||v>1))throw new Error('Invalid native metric');
 evidence.native={rows,metrics:['R@10','R@20','NDCG@10','NDCG@20'],seeds:'FFMSR 42/123; FedDCSR 42',protocol:'full ranking; fixed 5000 users/domain; one held-out positive, hence R@K = HR@K'};
 const slide=presentation.slides.add();slide.background.fill='#FFFFFF';
 const categories=rows.map(row=>`${row.method} ${row.run}`);
 for(let panel=0;panel<2;panel++){
  const x=panel?714:0;
  text(slide,panel?'(b) Full-ranking NDCG':'(a) Full-ranking recall',x,0,694,40,28,true);
  const series=[0,1].map(i=>({name:panel?`NDCG@${i?20:10}`:`R@${i?20:10}`,values:rows.map(r=>r.values[panel*2+i]),fill:[colors.blue,colors.coral][i],line:{fill:'none',width:0},dataLabelOverrides:rows.map((row,idx)=>({idx,text:row.values[panel*2+i].toFixed(4),showValue:false,position:'outEnd',textStyle:{...textStyle,fontSize:18}}))}));
  chart(slide,'bar',{position:{left:x,top:42,width:694,height:552},categories,series,
   barOptions:{direction:'bar',grouping:'clustered',gapWidth:62},
   xAxis:axis('',{majorGridlines:null,textStyle:{...textStyle,fontSize:20},position:'left'}),
   yAxis:axis(panel?'NDCG':'Recall',{min:0,max:panel?0.13:0.23,majorUnit:panel?0.03:0.05,numberFormatCode:'0.00',position:'bottom',textStyle:{...textStyle,fontSize:20}}),
   dataLabels:{showValue:false,position:'outEnd',textStyle:{...textStyle,fontSize:18}},hasLegend:true,legend:{position:'top',overlay:false,textStyle:{...textStyle,fontSize:23}}});
 }
 slide.speakerNotes.textFrame.setText('Native full-ranking FFMSR results are two-seed means (42,123) from outputs/external_baselines/ffmsr. FedDCSR retains its seed42 result. The nine rows exactly match Table 1b. Pipelines and item catalogs differ from the common-backbone experiment. Sampled uni99 records use separately named checkpoints and are not paired here.');
}
// 9: Joint training-evidence slices and an exact full-ranking gain decomposition.
{
 const joint=await readJson(path.join(here,'joint_evidence.json'));
 evidence.joint=joint;
 const slide=presentation.slides.add();slide.background.fill='#FFFFFF';
 const groups=['conflict','agree_cold','warm'];
 const cats=['Low / seen','Low / unseen','Frequency >5'];
 text(slide,'(a) Global frequency × client exposure',0,0,698,42,27,true);
 const ms=['fedpasa','pasa_no_routing','pasa_no_condition','pasa_no_anchor'];
 chart(slide,'bar',{position:{left:0,top:44,width:698,height:548},categories:cats,
  series:ms.map((m,i)=>({name:['Base','No routing','No FiLM','No anchor'][i],values:groups.map(g=>joint.cross_slices[m][g]['conflict_HR@20'].mean),fill:[colors.blue,colors.gray,colors.coral,colors.mint][i],line:{fill:'none',width:0}})),
  barOptions:{direction:'column',grouping:'clustered',gapWidth:70},
  xAxis:axis('Joint training-evidence group',{majorGridlines:null,textStyle:{...textStyle,fontSize:22}}),
  yAxis:axis('Full-ranking HR@20',{min:0,max:.04,majorUnit:.01,numberFormatCode:'0.00'}),
  hasLegend:true,legend:{position:'top',overlay:false,textStyle:{...textStyle,fontSize:21}}});
 text(slide,'(b) Where full-ranking gains originate',716,0,690,42,27,true);
 chart(slide,'bar',{position:{left:716,top:44,width:690,height:548},categories:cats,
  series:['cross','indomain'].map((stage,i)=>({name:['Cross-domain','In-domain'][i],values:groups.map(g=>100*joint.decomposition[stage].macro.contributions[g]),fill:[colors.purple,colors.gold][i],line:{fill:'none',width:0}})),
  barOptions:{direction:'column',grouping:'clustered',gapWidth:90},
  xAxis:axis('Joint training-evidence group',{majorGridlines:null,textStyle:{...textStyle,fontSize:22},tickLabelPosition:'low'}),
  yAxis:axis('Contribution to ΔHR@20 (pp)',{min:-.5,max:3,majorUnit:.5,numberFormatCode:'0.0'}),
  hasLegend:true,legend:{position:'top',overlay:false,textStyle:{...textStyle,fontSize:22}}});
 slide.speakerNotes.textFrame.setText('Source: joint_evidence.json generated by analyze_joint_evidence.py from retained v7 metrics_no_pap. Low means global training frequency <=5. Seen means at least one occurrence in the assigned client training histories, not frequent local consumption. (a) Six directions pooled within each seed, then five-seed mean. Support per seed C=236/210/244/222/244; U=264/290/256/278/256; W=29500. No final personalization. (b) Exact n_group/n_total weighted conditional HR difference: base minus target-only FedAvg-Adapter. Macro averages across six cross-domain directions or five in-domain datasets. All groups retained; negative bars are genuine offsets. Repeated target users across source directions are not independent observations. Complete HR and NDCG values are in tables_joint_slices.tex; decomposition in tables_joint_decomposition.tex. These are descriptive component-removal results, not a factorial interaction test.');
}
await fs.writeFile(path.join(here,'paper_figure_data.json'),JSON.stringify(evidence,null,2)+'\n');

const candidate=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidate);
// The chart facade has no logarithmic-axis option. Set the first scatter x-axis
// to log10 in the editable OOXML chart, preserving its actual frequency values.
const patch=String.raw`import zipfile,sys,xml.etree.ElementTree as E,os
p=sys.argv[1]
ns={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
q=lambda tag:'{'+ns[tag.split(':')[0]]+'}'+tag.split(':')[1]
with zipfile.ZipFile(p) as z: data={n:z.read(n) for n in z.namelist()}
for key in [n for n in data if '/charts/chart' in n and n.endswith('.xml')]:
 idx=int(key.rsplit('chart',1)[1].split('.')[0]);r=E.fromstring(data[key])
 for node in r.findall('.//c:smooth',ns):node.set('val','0')
 for node in r.findall('.//c:varyColors',ns):node.set('val','0')
 for node in r.findall('.//c:scatterStyle',ns):node.set('val','lineMarker')
 for j,ser in enumerate(r.findall('.//c:ser',ns)):
  smooth=ser.find('c:smooth',ns)
  if smooth is None:smooth=E.SubElement(ser,q('c:smooth'))
  smooth.set('val','0')
  # Preserve explicitly assigned colors of ablation bars.
  mark=ser.find('c:marker',ns)
  if mark is not None:
   colorNode=ser.find('c:spPr/a:solidFill/a:srgbClr',ns)
   if colorNode is None:colorNode=ser.find('c:spPr/a:ln/a:solidFill/a:srgbClr',ns)
   col=colorNode.get('val') if colorNode is not None else '487EB5'
   old=mark.find('c:spPr',ns)
   if old is not None:mark.remove(old)
   sp=E.SubElement(mark,q('c:spPr'));fill=E.SubElement(sp,q('a:solidFill'));E.SubElement(fill,q('a:srgbClr'),{'val':col});ln=E.SubElement(sp,q('a:ln'),{'w':'12700'});fill=E.SubElement(ln,q('a:solidFill'));E.SubElement(fill,q('a:srgbClr'),{'val':col})
  if idx==3 and j<6:
   for dl in ser.findall('c:dLbls/c:dLbl',ns):
    label=dl.find('c:dLblPos',ns)
    if label is None:
     label=E.Element(q('c:dLblPos'));first=next((k for k,n in enumerate(dl) if n.tag.rsplit('}',1)[1].startswith('show')),len(dl));dl.insert(first,label)
    label.set('val',['r','r','t','b','b','t'][j])
  if idx in [6,7,8,9]:
   labels=ser.find('c:dLbls',ns)
   if labels is not None:
    flags=['showLegendKey','showVal','showCatName','showSerName','showPercent','showBubbleSize']
    for owner in [labels]+labels.findall('c:dLbl',ns):
     for old in list(owner):
      if old.tag.rsplit('}',1)[1] in flags:owner.remove(old)
     at=next((k for k,node in enumerate(owner) if node.tag.rsplit('}',1)[1] in ['separator','showLeaderLines','leaderLines','extLst']),len(owner))
     for k,flag in enumerate(flags):owner.insert(at+k,E.Element(q('c:'+flag),{'val':'0'}))
   for dl in ser.findall('c:dLbls/c:dLbl',ns):
    label=dl.find('c:dLblPos',ns)
    if label is None:
     label=E.Element(q('c:dLblPos'));first=next((k for k,n in enumerate(dl) if n.tag.rsplit('}',1)[1].startswith('show')),len(dl));dl.insert(first,label)
    label.set('val','t')
   for dp in ser.findall('c:dPt',ns):
    marker=E.Element(q('c:marker'));E.SubElement(marker,q('c:symbol'),{'val':'circle'});E.SubElement(marker,q('c:size'),{'val':'11'});sp=E.SubElement(marker,q('c:spPr'));fill=E.SubElement(sp,q('a:solidFill'));E.SubElement(fill,q('a:srgbClr'),{'val':'FFFFFF'});ln=E.SubElement(sp,q('a:ln'),{'w':'25400'});fill=E.SubElement(ln,q('a:solidFill'));E.SubElement(fill,q('a:srgbClr'),{'val':col});dp.insert(1,marker)
 if idx==1:
  axes=r.findall('.//c:valAx',ns);a=next((a for a in axes if a.find('c:axPos',ns) is not None and a.find('c:axPos',ns).get('val')=='b'),axes[0]);sc=a.find('c:scaling',ns);sc.insert(0,E.Element(q('c:logBase'),{'val':'10'}))
 if r.find('.//c:barDir',ns) is not None and r.find('.//c:barDir',ns).get('val')=='bar':
  ca=r.find('.//c:catAx',ns);sc=ca.find('c:scaling',ns);ori=sc.find('c:orientation',ns)
  if ori is None:ori=E.SubElement(sc,q('c:orientation'))
  ori.set('val','maxMin');ca.find('c:axPos',ns).set('val','l')
  va=r.find('.//c:valAx',ns);va.find('c:axPos',ns).set('val','b')
  cr=va.find('c:crosses',ns)
  if cr is not None:cr.set('val','max')
 data[key]=E.tostring(r,encoding='utf-8',xml_declaration=True)
for key in [n for n in data if n.startswith('ppt/slides/slide') and n.endswith('.xml')]:
 r=E.fromstring(data[key]);changed=False
 for cell in r.findall('.//a:tcPr',ns):
  for old in list(cell):
   if old.tag in [q('a:'+edge) for edge in ['lnL','lnR','lnT','lnB']]:cell.remove(old)
  for i,edge in enumerate(['lnL','lnR','lnT','lnB']):
   ln=E.Element(q('a:'+edge),{'w':'6350'});fill=E.SubElement(ln,q('a:solidFill'));E.SubElement(fill,q('a:srgbClr'),{'val':'FFFFFF'});cell.insert(i,ln)
  changed=True
 if changed:data[key]=E.tostring(r,encoding='utf-8',xml_declaration=True)
tmp=p+'.tmp'
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in data.items():z.writestr(n,b)
os.replace(tmp,p)`;
execFileSync(python,['-c',patch,candidate]);
const finalPath=path.join(workspaceDir,'paper_www','figures','paper_evidence_editable_20261010.pptx');
const validatedPath=path.join(build,'validated',`experiment_figures_${Date.now()}.pptx`);
await fs.mkdir(path.dirname(validatedPath),{recursive:true});
await finalizePresentation({workspaceDir,candidatePath:candidate,finalPath:validatedPath,pythonExecutable:python,
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','13411200,5715000','--validate-heading-fit','--require-native-table-slide','5'],explicitTotalSlideCount:9,
 requiredNativeChartOwnerSlides:[1,2,3,4,6,7,8,9],requiredNativeTableOwnerSlides:[5],materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(build,`validation_${Date.now()}.json`)});
await fs.copyFile(validatedPath,finalPath);
console.log(finalPath);

