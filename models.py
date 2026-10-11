"""The graph network: one card per member, rounds of averaging along the lines, weight tables, training."""
import torch
from data import N, SIDE, INSTRUCTOR, ADMIN, average_table

torch.set_num_threads(1)
AVERAGE = torch.tensor(average_table(), dtype=torch.float32)
TRUE = torch.tensor([SIDE[m] for m in range(1, N + 1)])
CARD = 2          # numbers on a card
STEPS, RATE = 200, 0.02


def forward(cards, tables, rounds, average=AVERAGE):
    """Two scores per member. `rounds` of the tables are preceded by an averaging round; 0 = plain network."""
    h = cards
    for i, w in enumerate(tables):
        if i < rounds:
            h = average @ h              # every member: the average of the own card and the friends' cards
        h = h @ w                        # the same weight table for every member
        if i < len(tables) - 1:
            h = torch.tanh(h)            # squash between rounds
    return h


def start(seed, rounds=1, tables=None, n=N):
    """Random cards and random weight tables. The last table turns a card into two scores."""
    g = torch.Generator().manual_seed(seed)
    k = tables or max(rounds, 1)
    # one decimal, so that a round can be followed by hand
    cards = torch.nn.Parameter((torch.randn(n, CARD, generator=g) * 0.5).round(decimals=1))
    ws = [torch.nn.Parameter(torch.randn(CARD, CARD, generator=g) / CARD ** 0.5) for _ in range(k)]
    return cards, ws


def train(seed=0, rounds=1, names=(INSTRUCTOR, ADMIN), tables=None, steps=STEPS):
    """Train on the named members only. Returns a dict with everything the steps print."""
    cards, ws = start(seed, rounds, tables)
    first = cards.detach().clone()
    told = [m - 1 for m in names]
    opt = torch.optim.Adam([cards, *ws], lr=RATE)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(forward(cards, ws, rounds)[told], TRUE[told])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        scores = forward(cards, ws, rounds)
        guess = scores.argmax(1)
        others = [m for m in range(1, N + 1) if m not in names]
        wrong = [m for m in others if guess[m - 1] != TRUE[m - 1]]
        return {
            'right': len(others) - len(wrong), 'of': len(others), 'wrong': wrong,
            'told_right': int((guess[told] == TRUE[told]).sum()),
            'moved': int(((cards - first).abs().sum(1) > 0).sum()),      # cards that training changed
            'loss': float(loss), 'scores': scores.tolist(), 'guess': guess.tolist(),
            'first': first.tolist(), 'cards': cards.tolist(), 'tables': [w.tolist() for w in ws],
        }


def job(a):
    return train(**a)


def many(runs=50, **kw):
    """The same training from `runs` different random starts, on all processor cores."""
    from multiprocessing import Pool
    with Pool() as p:
        return p.map(job, [dict(seed=s, **kw) for s in range(runs)], chunksize=2)
