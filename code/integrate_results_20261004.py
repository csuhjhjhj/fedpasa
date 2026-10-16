"""Regenerate paper tables from completed v7 JSON records; never launches training.

Run with Python 3: python integrate_results_20261004.py
Inputs: outputs/www_v1/v7, validation/participation controls, native baseline JSON.
Outputs: paper_www active tables and paper_results_20261004.json for plotting.
"""
from pathlib import Path
import json
from statistics import mean, stdev
from functools import lru_cache
from itertools import product
from math import sqrt

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'outputs/www_v1'
V7 = OUT/'v7'
PAPER = ROOT.parent/'paper_www'
SEEDS = [42,123,456,789,2024]
THREE = SEEDS[:3]
DIRS = ['music_to_books','music_to_movies','books_to_movies','books_to_music','movies_to_books','movies_to_music']
DS = ['ml100k','digital_music','video_games','luxury_beauty','ml1m']
METHODS = ['fcf','scratch','fedprox','fedadam','transfr','l2sp','fedper','pfedme','fedmoe','fedpasa']
LABELS = ['FedMF (FCF)','FedAvg-Adapter','FedProx','FedAdam','TransFR control','L2-SP','FedPer','pFedMe-style','FedMoE','FedPASA']
K = ['s99_HR@20','s99_NDCG@20','HR@10','HR@20','NDCG@10','NDCG@20']

@lru_cache(None)
def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def record(method,direction,seed,base=False):
    group='cross';folder=method
    if method=='fedpasa' and not base:group,folder='ditto','pasa_ditto'
    if method=='pasa_nosrc':group='nosrc'
    if method=='decoupled':group,folder='fedecider','fedecider'
    p=V7/group/direction/folder/f'seed{seed}'/'results.json'
    r=read(p);a=r['args']
    assert a['seed']==seed and a['eval_sampled_negatives']==99 and a['eval_users']==5000,p
    assert (a['rounds'],a['local_steps'])==((8,30) if direction.endswith('books') else (12,60)),p
    key='metrics_fedmf' if method=='fcf' else 'metrics_ditto' if method=='pfedme' or (method=='fedpasa' and not base) else 'metrics_no_pap'
    m=r[key]
    assert all(isinstance(m[k],(float,int)) and 0<=m[k]<=1 for k in K),p
    return r,m

def cross(method,base=False,seeds=SEEDS):
    rows=[[record(method,d,s,base)[1] for s in seeds] for d in DIRS]
    values=[[mean(r[k] for r in rr) for k in K] for rr in rows]
    per_seed={k:[mean(rows[di][si][k] for di in range(6)) for si in range(len(seeds))] for k in K}
    return dict(values=values,mean=[mean(row[i] for row in values) for i in range(len(K))],
                sd=[stdev(per_seed[k]) for k in K],per_seed=per_seed,
                direction_sd=[[stdev(x[k] for x in rr) for k in K] for rr in rows],seeds=seeds)

def block(caption,label,headers,rows,spec=None,size='footnotesize',wide=False):
    width=r'\textwidth' if wide else r'\columnwidth'
    spec=spec or ('l'+'r'*(len(headers)-1))
    return '\n'.join([rf'\par\noindent\begin{{minipage}}{{{width}}}\centering\{size}',
      rf'\captionof{{table}}{{{caption}}}\label{{{label}}}',r'\setlength{\tabcolsep}{2.5pt}',
      rf'\begin{{tabular*}}{{\linewidth}}{{@{{\extracolsep{{\fill}}}}{spec}@{{}}}}',r'\toprule',
      ' & '.join(headers)+r' \\',r'\midrule',*[' & '.join(row)+r' \\' for row in rows],
      r'\bottomrule',r'\end{tabular*}',r'\end{minipage}\par\addvspace{6pt}'])+'\n'

def write(name,text): (PAPER/name).write_text(text,encoding='utf-8')
def fmt(x): return f'{x:.4f}'
def pair(v):return '/'.join(map(fmt,v[:2]))

def build():
    data={'directions':DIRS,'metrics':K,'cross':{},'indomain':{},'studies':{},'sensitivity':{},'validation':{},'native':[]}
    for m in METHODS: data['cross'][m]=cross(m)
    for m in ['fedpasa','pasa_no_routing','pasa_no_condition','pasa_no_anchor','pasa_const_routing','pasa_rand_routing','pasa_item_anchor','pasa_nosrc']:
        data['cross']['base' if m=='fedpasa' else m]=cross(m,base=True)
    data['cross']['decoupled']=cross('decoupled',seeds=THREE)
    data['uncertainty']={}
    for key in ['s99_HR@20','HR@20']:
        diffs=[a-b for a,b in zip(data['cross']['fedpasa']['per_seed'][key],data['cross']['fedmoe']['per_seed'][key])]
        mu=mean(diffs);delta=2.776445*stdev(diffs)/sqrt(5)
        exact=sum(abs(mean(s*d for s,d in zip(signs,diffs)))>=abs(mu)-1e-12 for signs in product([-1,1],repeat=5))/32
        data['uncertainty'][key]={'comparison':'FedPASA minus FedMoE','paired_seed_differences':diffs,'mean':mu,'t95':[mu-delta,mu+delta],'exact_signflip_p':exact}
    # Five-seed in-domain results, preserving the full-catalog evaluation.
    for m in ['fcf','scratch','fedprox','fedadam','fedper','pfedme','fedmoe','fedpasa','base']:
        vals=[]
        for ds in DS:
            folder='pasa_ditto' if m=='fedpasa' else 'fedpasa' if m=='base' else m
            key='metrics_fedmf' if m=='fcf' else 'metrics_ditto' if m in ['pfedme','fedpasa'] else 'metrics_no_pap'
            rr=[read(V7/'indomain'/ds/folder/f'seed{s}'/'results.json') for s in SEEDS]
            assert all(x['args']['rounds']==20 and x['args']['local_steps']==60 for x in rr)
            vals.append([mean(x[key][k] for x in rr) for k in ['HR@20','NDCG@20']])
        data['indomain'][m]={'values':vals,'mean':[mean(v[i] for v in vals) for i in range(2)]}
    # Validation-only selection. Test values are intentionally never read here.
    for g in ['validsweep','validsweep2']:
        grid=[]
        for d in sorted((OUT/g).iterdir()):
            ff=sorted(d.glob('seed*/results.json'))
            if not ff:continue
            rr=[read(f) for f in ff]
            assert sorted(x['args']['seed'] for x in rr)==sorted(THREE)
            grid.append({'setting':d.name,'HR20':mean(x['metrics_valid']['s99_HR@20'] for x in rr),
              'NDCG20':mean(x['metrics_valid']['s99_NDCG@20'] for x in rr)})
        data['validation'][g]=grid
    assert max(data['validation']['validsweep'],key=lambda x:x['HR20'])['setting']=='a0.85_k0.5'
    assert max(data['validation']['validsweep2'],key=lambda x:x['HR20'])['setting']=='l0.2_r0.3'
    for group,root in [('scalability',V7),('noniid',V7),('fixedcompute',OUT),('participation',OUT)]:
        rows=[]
        for d in sorted((root/group).iterdir(),key=lambda p:float(p.name.replace('clients','')) if p.name.startswith('clients') else p.name):
            if not d.is_dir():continue
            values=[];args=None
            for m in ['transfr','fedpasa']:
                rr=[read(d/m/f'seed{s}/results.json') for s in THREE];args=rr[0]['args']
                values.append([mean(x['metrics_no_pap'][k] for x in rr) for k in K[:2]])
            rows.append({'setting':d.name,'values':values,'config':{k:args[k] for k in ['rounds','local_steps','num_clients','client_participation','fedbat_alpha','fedbat_cold_power','fedbat_condition_scale','semantic_anchor_weight']}})
        data['studies'][group]=rows
    for name,field,default in [('alpha','fedbat_alpha',.85),('coldpower','fedbat_cold_power',.5),('condscale','fedbat_condition_scale',.2),('anchorw','semantic_anchor_weight',.3)]:
        pts=[]
        base=[record('fedpasa','music_to_movies',s,True)[0] for s in THREE]
        for p in sorted((V7/'sensitivity').glob(name+'*')):
            rr=[read(p/f'seed{s}/results.json') for s in THREE]
            for i,r in enumerate(rr):
                for key in ['rounds','local_steps','fedbat_alpha','fedbat_cold_power','fedbat_condition_scale','semantic_anchor_weight']:
                    if key!=field:assert r['args'][key]==base[i]['args'][key],(p,key)
            pts.append([rr[0]['args'][field],*[mean(x['metrics_no_pap'][k] for x in rr) for k in K[:2]]])
        pts.append([default,*[mean(x['metrics_no_pap'][k] for x in base) for k in K[:2]]])
        data['sensitivity'][name]={'default':default,'points':sorted(pts)}
    # Native FFMSR files retain their own model, task, and item catalogs.
    # Full-ranking records and uni99 records name different checkpoints: report separately.
    ffroot=ROOT/'outputs/external_baselines/ffmsr'
    for d in DIRS:
        name=d.replace('_to_','2');full=[];sample=[]
        for s in [42,123]:
            a=read(ffroot/name/f'seed{s}/result.json');b=read(ffroot/name/f'seed{s}/result_uni99.json')
            full.append(a['metrics']['test']);sample.append(b['metrics']['test'])
        data['native'].append({'method':'FFMSR','direction':d,'seeds':[42,123],
           'full':[mean(x[k] for x in full) for k in K[2:]],'sampled':[mean(x[k] for x in sample) for k in K[2:]]})
    fedd=read(ROOT/'external_baselines/results/feddcsr/seed42/FedDCSR_mbmv_seed42_results.json')
    for domain in ['Music','Books','Movies']:
        x=fedd['per_domain'][domain]
        data['native'].append({'method':'FedDCSR','direction':domain,'seeds':[42],
             'full':[x[k] for k in ['HR @10','HR @20','NDCG @10','NDCG @20']]})
    # Frequency slices retain zero-hit runs when their denominator is nonzero.
    data['buckets']=[]
    for bucket in ['zero','1_5','6_20','20plus']:
        rows=[];support=[]
        for d in DIRS:
            for s in SEEDS:
                mm=[record(m,d,s,True)[1] for m in ['transfr','fedmoe','fedpasa']]
                if all(x[f'freq_{bucket}_count']>0 for x in mm):rows.append(mm);support.append(mm[0][f'freq_{bucket}_count'])
        vals=[[mean(row[j].get(f's99freq_{bucket}_{k}',0) for row in rows) for k in ['HR@20','NDCG@20']] for j in range(3)]
        data['buckets'].append({'bucket':bucket,'values':vals,'runs':len(rows),'support':mean(support)})
    data['supplementary']=[]
    for noise in ['0.1','0.2','0.3']:
        vals=[]
        for method in ['transfr','fedpasa']:
            rr=[read(V7/'robustness'/('noise'+noise)/d/method/f'seed{s}/results.json') for d in DIRS for s in THREE]
            vals.append([mean(x['metrics_no_pap'][k] for x in rr) for k in K[:2]])
        data['supplementary'].append({'setting':'Update noise '+noise,'values':vals})
    for d,label in [('music_to_movies',r'LLaMA M$\to$V'),('books_to_movies',r'LLaMA B$\to$V'),('music_to_books',r'LLaMA M$\to$B')]:
        vals=[]
        for method in ['llama_transfr','llama_fedpasa']:
            rr=[read(V7/'encoder'/d/method/f'seed{s}/results.json') for s in THREE]
            vals.append([mean(x['metrics_no_pap'][k] for x in rr) for k in K[:2]])
        data['supplementary'].append({'setting':label,'values':vals})
    (ROOT/'paper_results_20261004.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    # Cross table, in-domain table and direction-wise SD retain the established appendix grid.
    headings=['Method',r'M$\to$B',r'M$\to$V',r'B$\to$V',r'B$\to$M',r'V$\to$B',r'V$\to$M','Mean']
    rows=[[lab,*[pair(v) for v in data['cross'][m]['values']],pair(data['cross'][m]['mean'])] for m,lab in zip(METHODS,LABELS)]
    top=block(r'Exact sampled HR@20/NDCG@20 for Figure~\ref{fig:cross_results} (five seeds). M/B/V: Music/Books/Movies. FedPASA includes final personalization.', 'tab:cross_main',headings,rows,wide=True)
    order=['fcf','scratch','fedprox','fedadam','fedper','pfedme','fedmoe','fedpasa']
    rows=[[dict(zip(METHODS,LABELS))[m],*[pair(v) for v in data['indomain'][m]['values']],pair(data['indomain'][m]['mean'])] for m in order]
    top+=block('In-domain full-ranking HR@20/NDCG@20, five-seed means. All eligible users are evaluated.', 'tab:indomain_main',['Method','ML-100K','Digital Music','Video Games','Luxury Beauty','ML-1M','Mean'],rows,wide=True,size='small')
    for j,name in enumerate(['hr','ndcg']):
        rows=[]
        for m,lab in [('transfr','TransFR control'),('fedmoe','FedMoE'),('base','FedPASA base'),('fedpasa','FedPASA')]:
            v=data['cross'][m]; rows.append([lab,*[rf'{v["values"][i][j]:.4f}$\pm${v["direction_sd"][i][j]:.4f}' for i in range(6)],rf'{v["mean"][j]:.4f}$\pm${v["sd"][j]:.4f}'])
        top+=block(('HR@20' if j==0 else 'NDCG@20')+' mean and sample SD. Mean-column SD is computed after averaging six directions within each seed.','tab:seed-'+name,headings,rows,wide=True)
    write('appendix_results_top.tex',top)
    # Main table contains actual full ranking plus native results, rather than cross-paper mean rankings.
    left=[r'\textbf{(a) Common-backbone full ranking (five seeds)}\\[3pt]',r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}',r'\toprule',r'Method & H10 & H20 & N10 & N20\\',r'\midrule']
    for m,lab in list(zip(METHODS[:-1],LABELS[:-1]))+[('base','FedPASA base'),('fedpasa',r'\textbf{FedPASA}')]:
        left.append(' & '.join([lab,*map(fmt,data['cross'][m]['mean'][2:])])+r' \\')
    left+=[r'\midrule','Decoupled-style$^*$ & '+' & '.join(map(fmt,data['cross']['decoupled']['mean'][2:]))+r' \\',r'\bottomrule\end{tabular*}',r'\par\smallskip\raggedright $^*$Three seeds; frozen LLaMA features and an adapter aggregation control inspired by FeDecider, not its official implementation.']
    short={d:l for d,l in zip(DIRS,['M--B: B','M--V: V','B--V: V','M--B: M','B--V: B','M--V: M'])}
    right=[r'\textbf{(b) Native author-code full ranking}\\[3pt]',r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrrr@{}}',r'\toprule',r'Method & Run: domain & H10 & H20 & N10 & N20\\',r'\midrule']
    for row in data['native']:
        right.append(' & '.join([row['method'],short.get(row['direction'],row['direction']),*map(fmt,row['full'])])+r' \\')
    right+=[r'\bottomrule\end{tabular*}',r'\par\smallskip\raggedright FFMSR: seeds 42 and 123. FedDCSR: seed 42. Native catalogs, sequence inputs and budgets differ from panel (a).']
    text=r'''\begin{table*}[t]\centering\footnotesize
\caption{Full-ranking evidence in the main comparison. H/N denote HR/NDCG at 10 or 20. (a) Six-direction means under common features and target budgets. (b) Released systems retain their native pipelines; M/B/V denote Music/Books/Movies. These panels do not form a matched cross-system ranking.}
\label{tab:external-main}\label{tab:native-reproductions}\label{tab:full-ranking-check}
\setlength{\tabcolsep}{2pt}
\begin{minipage}[t]{.47\textwidth}\centering
'''+ '\n'.join(left)+r'''
\end{minipage}\hfill
\begin{minipage}[t]{.51\textwidth}\centering
'''+ '\n'.join(right)+r'''
\end{minipage}
\end{table*}
'''
    write('tables_external_main.tex',text)
    variants=[('base','FedPASA base'),('pasa_no_routing','Without routing'),('pasa_no_condition','Without FiLM'),('pasa_no_anchor','Without AutoAnchor'),('pasa_const_routing','Constant routing'),('pasa_rand_routing','Random routing'),('pasa_item_anchor','Per-item anchor'),('pasa_nosrc','Source-free base'),('fedmoe','Learned-gate FedMoE'),('fedpasa','Final personalization')]
    rows=[[lab,*map(fmt,data['cross'][m]['mean'][:2])] for m,lab in variants]
    write('tables_ablation_revised.tex',block(r'Cross-domain controls (six directions, five seeds). First seven rows: Figure~\ref{fig:ablation}; only the last row includes final personalization.','tab:ablation',['Variant','HR@20','NDCG@20'],rows))
    rows=[]
    for b in data['buckets']:
        for j,m in enumerate(['TransFR','FedMoE','FedPASA base']):rows.append([{'zero':'Unseen','1_5':'1--5','6_20':'6--20','20plus':'>20'}[b['bucket']] if j==0 else '',m,*map(fmt,b['values'][j])])
    write('appendix_results_bottom.tex',r'\input{tables_ablation_revised}\newpage'+'\n'+block(r'Fine-grained frequency slices (99-negative ranking; unpersonalized models).','tab:buckets',['Frequency','Method','HR@20','NDCG@20'],rows)+r'\noindent Common nonempty runs: 20 unseen (mean support 13 users); 30 otherwise. Zero-hit runs are retained.\par'+'\n')
    # Compact full validation grids replace the duplicated full-ranking summary in Appendix D.
    v1={x['setting']:x['HR20'] for x in data['validation']['validsweep']};v2={x['setting']:x['HR20'] for x in data['validation']['validsweep2']}
    leftrows=[[str(a),*[fmt(v1[f'a{a}_k{k}']) for k in [.5,.7,1.0]]] for a in [.4,.55,.7,.85]]
    rightrows=[[str(l),*[fmt(v2[f'l{l}_r{r}']) for r in [.1,.2,.3]]] for l in [.2,.4,.6]]
    def grid(headers,rows):return '\n'.join([r'\begin{tabular}{lrrr}\toprule',' & '.join(headers)+r'\\\midrule',*[' & '.join(row)+r'\\' for row in rows],r'\bottomrule\end{tabular}'])
    text=r'''\begin{minipage}{\textwidth}\centering\small
\captionof{table}{Validation-only selection on Music$\to$Movies (mean sampled HR@20, seeds 42/123/456; eight rounds, 30 local steps). Stage 1 fixes $\lambda=.4,\rho=.2$; Stage 2 fixes the selected $\alpha=.85,\kappa=.5$. The final choice is $(.85,.5,.2,.3)$.}\label{tab:validation-grid}
\begin{minipage}{.48\textwidth}\centering
'''+grid([r'$\alpha\backslash\kappa$','.5','.7','1.0'],leftrows)+r'\end{minipage}\hfill\begin{minipage}{.48\textwidth}\centering'+'\n'+grid([r'$\lambda\backslash\rho$','.1','.2','.3'],rightrows)+r'\end{minipage}\end{minipage}\par\addvspace{6pt}'+'\n'
    write('appendix_fullranking_top.tex',text)
    # Every supplementary condition has an exact pair of values in one table.
    rows=[]
    labels={'clients5':'5 clients','clients20':'20 clients','clients50':'50 clients','dirichlet_activity_a0.1':'Activity .1','dirichlet_activity_a0.5':'Activity .5','dirichlet_popularity_a0.5':'Popularity .5','rate0.2':'Participation .2','rate0.5':'Participation .5'}
    for g in ['scalability','noniid','fixedcompute','participation']:
        for x in data['studies'][g]:rows.append([('Fixed: ' if g=='fixedcompute' else '')+labels[x['setting']],*[pair(z) for z in x['values']]])
    rows.extend([[x['setting'],*[pair(v) for v in x['values']]] for x in data['supplementary']])
    supplement=block(r'HR@20/NDCG@20 (three seeds, base model). Client, fixed-compute and participation rows match Figure~\ref{fig:robustness}. Noise averages six directions; encoder rows use the indicated direction.','tab:external-conditions',['Setting','TransFR','FedPASA base'],rows)
    anchorrows=[[lab,*map(fmt,data['cross'][m]['mean'][:2])] for m,lab in [('pasa_no_anchor','None'),('base','Catalog gate'),('pasa_item_anchor','Per-item gate')]]
    supplement+=block('Anchor granularity (five-seed cross-domain means).','tab:anchor',['Anchoring','HR@20','NDCG@20'],anchorrows)
    rows=[]
    for domain in ['cross','ml100k']:
        for key,label in [('drift_cos','all'),('drift_cos_cold','cold')]:
            vv=[]
            for m in ['transfr','pasa_no_anchor','fedpasa']:
                if domain=='cross': mm=[record(m,d,s,True)[1] for d in DIRS for s in SEEDS]
                else:
                    # No new in-domain no-anchor sweep: do not reuse old-settings values.
                    if m=='pasa_no_anchor':vv.append(None);continue
                    folder='scratch' if m=='transfr' else m
                    mm=[read(V7/'indomain/ml100k'/folder/f'seed{s}/results.json')['metrics_no_pap'] for s in SEEDS]
                vv.append(mean(x[key] for x in mm))
            rows.append([('Cross' if domain=='cross' else 'ML-100K')+' '+label,*['--' if x is None else fmt(x) for x in vv]])
    supplement+=block('Cosine distance from the frozen prior. Reference: TransFR cross-domain, FedAvg in-domain. --: no matched new run.','tab:drift',['Slice','Reference','No anchor','Base'],rows)
    supplement+=r'\noindent Fixed-compute and participation controls use eight rounds with $\alpha=.55$, $\kappa=.7$, $\lambda=.4$, $\rho=.2$. Total steps per round are 150/160/150 at 5/20/50 clients. The 100\% participation point reuses the five-client control. Other rows use the selected configuration.\par'+'\n'
    write('appendix_settings_right.tex',supplement)
    rows=[]
    symbols=[r'$\alpha$',r'$\kappa$',r'$\lambda$',r'$\rho$']
    for (name,x),symbol in zip(data['sensitivity'].items(),symbols):
        for i,p in enumerate(x['points']):rows.append([symbol if i==0 else '',str(p[0]),pair(p[1:])+(' (d)' if p[0]==x['default'] else '')])
    write('sensitivity_values_revised.tex',block(r'HR@20/NDCG@20 for Figure~\ref{fig:sensitivity} (Music$\to$Movies, three seeds). (d): selected default.','tab:sensitivity-values',['Factor','Value','HR@20/NDCG@20'],rows))
    print('Cross means:',{m:[round(x,6) for x in d['mean'][:2]] for m,d in data['cross'].items()})
    print('In-domain:',{m:x['mean'] for m,x in data['indomain'].items()})
    print('Completed source-backed tables and plotting data.')

if __name__=='__main__':build()
