"""Build black-and-white booktabs from the existing verified result summary.

Run after integrate_results_20261004.py. No experiments or values are changed.
Best values are bold; second-best values are underlined within each column.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent / 'paper_www'
D = json.loads((ROOT / 'paper_results_20261004.json').read_text())
METHODS = ['fcf','scratch','fedprox','fedadam','transfr','l2sp','fedper','pfedme','fedmoe','fedpasa']
LABELS = ['FedMF','FedAvg-Adapter','FedProx','FedAdam','TransFR control','L2-SP','FedPer','pFedMe-style','FedMoE','FedPASA']
DIRS = [r'M$\to$B',r'M$\to$V',r'B$\to$V',r'B$\to$M',r'V$\to$B',r'V$\to$M','Mean']

def tabular(labels, headings, matrix):
    ranks = [sorted(set(row[i] for row in matrix), reverse=True) for i in range(len(headings))]
    lines = [r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l'+'r'*len(headings)+'@{}}',
             r'\toprule',r'\textbf{Method} & '+' & '.join(r'\textbf{'+h+'}' for h in headings)+r' \\',r'\midrule']
    for i, (label,row) in enumerate(zip(labels,matrix)):
        if label=='FedPASA': lines.append(r'\midrule')
        values=[]
        for j,v in enumerate(row):
            s=f'{v:.4f}'
            if v==ranks[j][0]:s=r'\textbf{'+s+'}'
            elif len(ranks[j])>1 and v==ranks[j][1]:s=r'\underline{'+s+'}'
            values.append(s)
        lines.append((r'\textbf{FedPASA}' if label=='FedPASA' else label)+' & '+' & '.join(values)+r' \\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular*}'])

def cross_table(indices,caption,label,environment):
    lines=[r'\begin{'+environment+r'}[t]' if environment=='table*' else r'\begin{minipage}{\textwidth}',
           r'\centering\small',r'\caption'+('of{table}' if environment=='minipage' else '')+'{'+caption+'}',
           r'\label{'+label+'}',r'\setlength{\tabcolsep}{3pt}',r'\renewcommand{\arraystretch}{1.13}']
    for i,k in enumerate(indices):
        lines += [r'\textbf{('+chr(97+i)+') '+D['metrics'][k].removeprefix('s99_')+r'}\par\smallskip',
                  tabular(LABELS,DIRS,[[r[k] for r in D['cross'][m]['values']]+[D['cross'][m]['mean'][k]] for m in METHODS])]
        if i==0:lines.append(r'\par\medskip')
    lines.append(r'\end{'+environment+'}')
    return '\n'.join(lines)+'\n'

def main():
    (PAPER/'tables_cross_monochrome.tex').write_text(cross_table([0,1],
        r'Cross-domain sampled ranking (five-seed means). M/B/V: Music/Books/Movies. '
        r'Bold and underline denote the best and second-best values. FedPASA includes final personalization.',
        'tab:cross_main','table*'),encoding='utf-8')
    (PAPER/'appendix_fullranking_directions.tex').write_text(cross_table([3,5],
        r'Direction-wise full-ranking HR@20 and NDCG@20 (five seeds), complementing Table~\ref{tab:external-main}. '
        r'Bold/underline denote the best/second-best values; FedPASA includes final personalization.',
        'tab:full-directions','minipage'),encoding='utf-8')
    methods=['fcf','scratch','fedprox','fedadam','fedper','pfedme','fedmoe','fedpasa']
    labels=[LABELS[METHODS.index(m)] for m in methods]
    # Pair both metrics without coloring cells; emphasize only the winning metric.
    hs=['ML-100K','Digital Music','Video Games','Luxury Beauty','ML-1M','Mean']
    cube=[[x for x in D['indomain'][m]['values']]+[D['indomain'][m]['mean']] for m in methods]
    lines=[r'\begin{table*}[t]\centering\small',r'\caption{In-domain full-ranking HR@20/NDCG@20 (five-seed means). '
           r'All eligible users are evaluated. Bold values mark the best result for each metric.}',
           r'\label{tab:indomain_main}',r'\setlength{\tabcolsep}{2pt}',r'\renewcommand{\arraystretch}{1.05}',
           r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrr@{}}',r'\toprule',
           r'\textbf{Method} & '+' & '.join(r'\textbf{'+h+'}' for h in hs)+r' \\',r'\midrule']
    for i,label in enumerate(labels):
        if label=='FedPASA':lines.append(r'\midrule')
        cells=[]
        for j in range(6):
            pair=[]
            for k in range(2):
                v=cube[i][j][k];s=f'{v:.4f}'
                pair.append(r'\textbf{'+s+'}' if v==max(row[j][k] for row in cube) else s)
            cells.append('/'.join(pair))
        lines.append((r'\textbf{FedPASA}' if label=='FedPASA' else label)+' & '+' & '.join(cells)+r' \\')
    lines += [r'\bottomrule\end{tabular*}',r'\end{table*}']
    (PAPER/'tables_indomain_main.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Keep seed dispersion tables in the supplement, without repeating main tables.
    p=PAPER/'appendix_results_top.tex'; text=p.read_text(encoding='utf-8')
    blocks=text.split(r'\par\noindent\begin{minipage}')
    retained=[r'\par\noindent\begin{minipage}'+b for b in blocks[1:] if r'\label{tab:seed-' in b]
    p.write_text(''.join(retained),encoding='utf-8')
    bottom=(PAPER/'appendix_results_bottom.tex').read_text(encoding='utf-8')
    (PAPER/'tables_frequency_slices.tex').write_text(bottom.split(r'\newpage',1)[1],encoding='utf-8')
    print('Generated monochrome main tables and full-ranking direction tables.')

if __name__=='__main__':main()
