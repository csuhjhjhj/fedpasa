import json
from pathlib import Path
from statistics import mean

def collect(base, variant, in_domain):
    vals = []
    for res in Path(base).rglob('results.json'):
        s = str(res).replace('\\', '/')
        if f'/{variant}/' not in s:
            continue
        if in_domain and '/in_' not in s:
            continue
        if not in_domain and '/in_' in s:
            continue
        m = json.load(open(res))['metrics_no_pap']
        if m.get('HR@20') is not None:
            vals.append(m)
    return vals

print('=== cross (s99 HR@20 / s99 NDCG@20)')
for label, base, variant in [('fedpasa', 'outputs/www_v1/cross', 'fedpasa'),
                             ('pasa_ll', 'outputs/www_v1/enhance', 'pasa_ll'),
                             ('pasa_cd', 'outputs/www_v1/enhance', 'pasa_cd'),
                             ('pasa_both', 'outputs/www_v1/enhance', 'pasa_both')]:
    vals = collect(base, variant, False)
    if vals:
        print(f'  {label:<12} {mean(m["s99_HR@20"] for m in vals):.4f}/{mean(m["s99_NDCG@20"] for m in vals):.4f} (n={len(vals)})')
print('=== indomain (full HR@20 / NDCG@20)')
for label, base, variant in [('fedpasa', 'outputs/www_v1/indomain', 'fedpasa'),
                             ('pasa_ll', 'outputs/www_v1/enhance', 'pasa_ll'),
                             ('pasa_cd', 'outputs/www_v1/enhance', 'pasa_cd'),
                             ('pasa_both', 'outputs/www_v1/enhance', 'pasa_both')]:
    vals = collect(base, variant, True)
    if vals:
        print(f'  {label:<12} {mean(m["HR@20"] for m in vals):.4f}/{mean(m["NDCG@20"] for m in vals):.4f} (n={len(vals)})')
print()
print('=== per-dataset/direction breakdown for pasa_ll vs fedpasa (HR@20)')
for ds_dir, base2 in [('cross', 'outputs/www_v1/cross')]:
    pass
for group in ['music_to_books','music_to_movies','books_to_movies','books_to_music','movies_to_books','movies_to_music']:
    a = [json.load(open(r))['metrics_no_pap']['s99_HR@20'] for r in Path(f'outputs/www_v1/cross/{group}/fedpasa').glob('seed*/results.json')]
    b = [json.load(open(r))['metrics_no_pap']['s99_HR@20'] for r in Path(f'outputs/www_v1/enhance/{group}/pasa_ll').glob('seed*/results.json')]
    if a and b:
        print(f'  {group:<16} fedpasa {mean(a):.4f} -> pasa_ll {mean(b):.4f} ({mean(b)-mean(a):+.4f})')
for ds in ['ml100k','ku','luxury_beauty','ml1m']:
    a = [json.load(open(r))['metrics_no_pap']['HR@20'] for r in Path(f'outputs/www_v1/indomain/{ds}/fedpasa').glob('seed*/results.json')]
    b = [json.load(open(r))['metrics_no_pap']['HR@20'] for r in Path(f'outputs/www_v1/enhance/in_{ds}/pasa_ll').glob('seed*/results.json')]
    if a and b:
        print(f'  in_{ds:<13} fedpasa {mean(a):.4f} -> pasa_ll {mean(b):.4f} ({mean(b)-mean(a):+.4f})')

