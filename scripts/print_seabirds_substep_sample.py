import json
from pathlib import Path
p = Path('build/seabirds_substep_detailed.json')
if not p.exists():
    print('missing')
else:
    d = json.load(p.open())
    print('entries', len(d))
    for i in range(min(3, len(d))):
        dd = d[i]
        print(i, 'month', dd['month'], 'biom', dd['biomass'], 'k1', dd['k1'], 'total_delta', dd['total_delta'])
