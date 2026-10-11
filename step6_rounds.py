"""Step 6. More rounds reach more of the club. How many rounds are good?"""
from statistics import mean
import torch
from models import many, start, AVERAGE
from step3_round import widest

ROUNDS = [1, 2, 4, 8, 16, 32, 64]


def blur():
    """Averaging only, no weights and no training: the largest distance between two cards after each round."""
    cards = start(0)[0].detach()
    out = {0: widest(cards)}
    for k in range(1, max(ROUNDS) + 1):
        cards = AVERAGE @ cards
        if k in ROUNDS:
            out[k] = widest(cards)
    return out


def summary(runs):
    return {'mean': mean(r['right'] for r in runs), 'min': min(r['right'] for r in runs),
            'max': max(r['right'] for r in runs),
            'told_both': sum(r['told_right'] == 2 and r['loss'] < 0.1 for r in runs), 'runs': len(runs),
            # the largest gap between the scores of any two members, in a typical run
            'score_gap': sorted(max(max(s[k] for s in r['scores']) - min(s[k] for s in r['scores']) for k in (0, 1))
                                for r in runs)[len(runs) // 2]}


def run():
    return {'rounds': {k: summary(many(rounds=k)) for k in ROUNDS}, 'blur': blur(),
            # the check: 64 weight tables as well, but only the first one is preceded by an averaging round
            'control': summary(many(rounds=1, tables=64))}


if __name__ == '__main__':
    r = run()
    print('rounds   right of 32   learned the 2 told names   largest gap between two members\' scores')
    for k, s in r['rounds'].items():
        print(f"{k:4d}     {s['mean']:5.1f} ({s['min']}-{s['max']})    {s['told_both']:2d} of {s['runs']} runs"
              f"              {s['score_gap']:.4f}")
    c = r['control']
    print(f"check, 64 weight tables with 1 averaging round: {c['mean']:.1f} right, "
          f"learned the 2 names in {c['told_both']} of {c['runs']} runs")
    print('averaging only, largest distance between two cards:',
          ', '.join(f"{k} rounds {v:.4f}" for k, v in r['blur'].items()))
