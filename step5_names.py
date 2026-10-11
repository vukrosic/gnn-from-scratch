"""Step 5. Does it matter which two names we tell it? Every pair with one member from each side."""
from multiprocessing import Pool
from statistics import mean
from data import N, SIDE, INSTRUCTOR, ADMIN, FRIENDS
from models import job

RUNS = 5


def run():
    pairs = [(a, b) for a in range(1, N + 1) if SIDE[a] == 0 for b in range(1, N + 1) if SIDE[b] == 1]
    with Pool() as p:
        res = p.map(job, [dict(seed=s, names=pr) for pr in pairs for s in range(RUNS)], chunksize=8)
    score = {pr: mean(r['right'] for r in res[i * RUNS:(i + 1) * RUNS]) for i, pr in enumerate(pairs)}
    order = sorted(pairs, key=score.get)
    return {'pairs': len(pairs), 'mean': mean(score.values()), 'leaders': score[(INSTRUCTOR, ADMIN)],
            'worst': [order[0], score[order[0]]], 'best': [order[-1], score[order[-1]]],
            'worst_friends': [len(FRIENDS[m]) for m in order[0]],
            'below_24': sum(v < 24 for v in score.values()), 'at_least_30': sum(v >= 30 for v in score.values()),
            'all': sorted(score.values()), 'runs': RUNS}


if __name__ == '__main__':
    r = run()
    print(f"{r['pairs']} pairs of names, {r['runs']} runs each, right on the other 32")
    print(f"average over all pairs: {r['mean']:.1f}")
    print(f"the two leaders: {r['leaders']:.1f}")
    print(f"best pair {r['best'][0]}: {r['best'][1]:.1f};  worst pair {r['worst'][0]}: {r['worst'][1]:.1f}")
    print(f"pairs with 30 or more right: {r['at_least_30']};  pairs below 24: {r['below_24']}")
