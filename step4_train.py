"""Step 4. The same network with one averaging round before the weight table. Told the same two names."""
from collections import Counter
from statistics import mean
import torch
from data import N, LINES, SIDE, FRIENDS, average_table
from models import many, train, forward


def renumbered(seed=0):
    """Give every member a new number, run the trained network again, compare each member's scores."""
    r = train(seed)
    new = torch.randperm(N, generator=torch.Generator().manual_seed(1)).tolist()   # member m+1 becomes new[m]+1
    lines = [(new[a - 1] + 1, new[b - 1] + 1) for a, b in LINES]
    cards = torch.zeros(N, 2)
    cards[new] = torch.tensor(r['cards'])
    table = torch.tensor(average_table(lines), dtype=torch.float32)
    scores = forward(cards, [torch.tensor(w) for w in r['tables']], 1, table)
    return float((scores[new] - torch.tensor(r['scores'])).abs().max())


def run():
    runs = many(rounds=1)
    right = [r['right'] for r in runs]
    wrong = Counter(m for r in runs for m in r['wrong'])
    ex = train(0)
    still = [m for m in range(1, N + 1) if ex['cards'][m - 1] == ex['first'][m - 1]]
    return {'mean': mean(right), 'min': min(right), 'max': max(right), 'of': runs[0]['of'], 'runs': len(runs),
            'moved': ex['moved'], 'still': still, 'wrong': dict(wrong.most_common()),
            'nine_wrong': wrong[9], 'nine_friends': FRIENDS[9], 'nine_sides': [SIDE[f] for f in FRIENDS[9]],
            'hist': dict(Counter(right)), 'example': ex, 'renumbered': renumbered()}


if __name__ == '__main__':
    r = run()
    print(f"one averaging round, told 2 names, {r['runs']} different random starts")
    print(f"right on the other {r['of']}: {r['mean']:.1f} on average (from {r['min']} to {r['max']})")
    print(f"cards that training changed: {r['moved']} of 34; unchanged: members {r['still']}")
    print(f"how often each member was wrong in {r['runs']} runs: {r['wrong']}")
    print(f"member 9 has friends {r['nine_friends']}; {sum(r['nine_sides'])} of them stayed with the administrator")
    print(f"renumber all 34 members and run again: largest change in any score {r['renumbered']:.7f}")
