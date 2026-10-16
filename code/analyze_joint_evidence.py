"""Recompute joint evidence slices from retained v7 evaluations (no training).

Input: outputs/www_v1/v7/{cross,indomain}/.../results.json.
Output: joint_evidence.json. Missing hit keys in nonempty slices denote zero
hits in the evaluator; empty slices are never treated as measured zero rates.
All rates below pool counts within a seed before averaging the five seeds.
"""
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
V7 = ROOT / 'outputs/www_v1/v7'
SEEDS = [42, 123, 456, 789, 2024]
DIRS = ['music_to_books', 'music_to_movies', 'books_to_movies',
        'books_to_music', 'movies_to_books', 'movies_to_music']
DATASETS = ['ml100k', 'digital_music', 'video_games', 'luxury_beauty', 'ml1m']
SLICES = ['conflict', 'agree_cold', 'warm']
METHODS = ['transfr', 'scratch', 'fedmoe', 'fedpasa', 'pasa_no_routing',
           'pasa_no_condition', 'pasa_no_anchor']

def read(stage, group, method, seed):
    p = V7 / stage / group / method / f'seed{seed}' / 'results.json'
    r = json.loads(p.read_text(encoding='utf-8-sig'))
    assert r['args']['seed'] == seed
    assert r['args']['cold_item_threshold'] == 5
    m = r['metrics_no_pap']
    assert sum(m[f'conflict_{g}_count'] for g in SLICES) == m['count']
    # Verify that serialized conditional means reconstruct the global metric.
    for metric in ['HR@20', 'NDCG@20']:
        for sampled in [False, True]:
            pref = 's99conflict' if sampled else 'conflict'
            key = ('s99_' if sampled else '') + metric
            total = sum(m[f'conflict_{g}_count'] * m.get(f'{pref}_{g}_{metric}', 0)
                        for g in SLICES) / m['count']
            assert abs(total - m[key]) < 1e-10, (p, key, total, m[key])
    return m

def pooled(stage, groups, method):
    rows = [[read(stage, g, method, s) for g in groups] for s in SEEDS]
    result = {}
    for g in SLICES:
        ns = [sum(m[f'conflict_{g}_count'] for m in rr) for rr in rows]
        assert min(ns) > 0
        v = {'support_per_seed': ns, 'mean_support': mean(ns)}
        for prefix in ['conflict', 's99conflict']:
            for metric in ['HR@20', 'NDCG@20']:
                vals = [sum(m[f'conflict_{g}_count'] * m.get(f'{prefix}_{g}_{metric}', 0)
                            for m in rr) / n for rr, n in zip(rows, ns)]
                v[prefix + '_' + metric] = {'mean': mean(vals), 'sd': stdev(vals), 'seeds': vals}
        result[g] = v
    return result

def decomposition(stage, groups):
    """Macro average of exact within-benchmark full-ranking gain decompositions."""
    output = {}
    for group in groups:
        pieces = {g: [] for g in SLICES}
        baseline, base, ndcg_b, ndcg_f, shares = [], [], [], [], {g: [] for g in SLICES}
        activity = {g: {'n': [], 'baseline': [], 'base': []} for g in ['le10', '11_50', '50plus']}
        for seed in SEEDS:
            b, f = read(stage, group, 'scratch', seed), read(stage, group, 'fedpasa', seed)
            baseline.append(b['HR@20']); base.append(f['HR@20'])
            ndcg_b.append(b['NDCG@20']); ndcg_f.append(f['NDCG@20'])
            for g in SLICES:
                n = f[f'conflict_{g}_count']
                assert n == b[f'conflict_{g}_count']
                w = n / f['count']; shares[g].append(w)
                pieces[g].append(w * (f.get(f'conflict_{g}_HR@20', 0) - b.get(f'conflict_{g}_HR@20', 0)))
            assert abs(sum(pieces[g][-1] for g in SLICES) - (f['HR@20'] - b['HR@20'])) < 1e-10
            for a in activity:
                n = f[f'uact_{a}_count']; assert n == b[f'uact_{a}_count']
                activity[a]['n'].append(n)
                activity[a]['baseline'].append(b.get(f'uact_{a}_HR@20', 0) if n else None)
                activity[a]['base'].append(f.get(f'uact_{a}_HR@20', 0) if n else None)
        output[group] = {'baseline': mean(baseline), 'base': mean(base),
                         'ndcg_baseline': mean(ndcg_b), 'ndcg_base': mean(ndcg_f),
                         'gain': mean(base) - mean(baseline),
                         'contributions': {g: mean(v) for g, v in pieces.items()},
                         'shares': {g: mean(v) for g, v in shares.items()}, 'activity': activity}
    output['macro'] = {key: mean(output[g][key] for g in groups)
                       for key in ['baseline', 'base', 'gain', 'ndcg_baseline', 'ndcg_base']}
    output['macro']['contributions'] = {s: mean(output[g]['contributions'][s] for g in groups) for s in SLICES}
    output['macro']['shares'] = {s: mean(output[g]['shares'][s] for g in groups) for s in SLICES}
    return output

def main():
    cross = {m: pooled('cross', DIRS, m) for m in METHODS}
    for m in METHODS:
        for g in SLICES:
            assert cross[m][g]['support_per_seed'] == cross['fedpasa'][g]['support_per_seed']
    data = {'seeds': SEEDS, 'directions': DIRS, 'datasets': DATASETS,
            'definition': 'Global training frequency <=5 and client training count >0: conflict; <=5 and local count=0: agree_cold; global count>5: warm. Local exposure is not high local frequency.',
            'cross_slices': cross,
            'decomposition': {'cross': decomposition('cross', DIRS), 'indomain': decomposition('indomain', DATASETS)}}
    (ROOT / 'joint_evidence.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    write_tables(data)
    for g in SLICES:
        print(g, 'support', cross['fedpasa'][g]['support_per_seed'])
        for m in METHODS:
            print(m, *(round(cross[m][g][p + '_' + k]['mean'], 5)
                       for p in ['s99conflict', 'conflict'] for k in ['HR@20', 'NDCG@20']))
    for stage, rows in data['decomposition'].items():
        print(stage)
        for g, row in rows.items():
            print(g, {k: row[k] for k in ['baseline', 'base', 'gain', 'contributions', 'shares']})

def write_tables(data):
    from integrate_results_20261004 import block
    labels = [('fedpasa', 'FedPASA base'), ('pasa_no_routing', 'Without routing'),
              ('pasa_no_condition', 'Without FiLM'), ('pasa_no_anchor', 'Without anchor')]
    rows = []
    for method, label in labels:
        cells = []
        for s in SLICES:
            v = data['cross_slices'][method][s]
            cells.append('/'.join(f"{v['conflict_' + k]['mean']:.4f}" for k in ['HR@20', 'NDCG@20']))
        rows.append([label] + cells)
    caption = (r'Joint evidence slices: full HR@20/NDCG@20. '
               r'C: low-frequency/local-seen; U: low-frequency/local-unseen; W: frequency $>5$. '
               r'Mean supports per seed: 231.2/268.8/29,500 direction--user evaluations; five seeds.')
    (ROOT.parent/'paper_www/tables_joint_slices.tex').write_text(
        block(caption, 'tab:joint', ['Variant', 'C', 'U', 'W'], rows), encoding='utf-8')
    rows = []
    labels = [('cross','macro','Cross mean'), ('indomain','ml100k','ML-100K'),
              ('indomain','digital_music','Digital Music'), ('indomain','video_games','Video Games'),
              ('indomain','luxury_beauty','Luxury Beauty'), ('indomain','ml1m','ML-1M'),
              ('indomain','macro','In-domain mean')]
    for stage, group, label in labels:
        r = data['decomposition'][stage][group]
        rows.append([label] + [f'{100*r["contributions"][s]:+.3f}' for s in SLICES]
                    + [f'{100*r["gain"]:+.3f}'])
    caption = (r'Full-ranking HR@20 gain decomposition, in percentage points: '
               r'base minus FedAvg-Adapter. C/U/W follow Table~\ref{tab:joint}.')
    (ROOT.parent/'paper_www/tables_joint_decomposition.tex').write_text(
        block(caption, 'tab:joint-decomp', ['Benchmark','C','U','W',r'$\Delta$'], rows), encoding='utf-8')

if __name__ == '__main__':
    main()
