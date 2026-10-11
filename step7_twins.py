"""Step 7. Two limits. Drawings of six atoms of the same kind, so every atom starts with the same card.
A: a chain of six against a ring of six.  B: a ring of six against two rings of three."""
import torch
from data import average_table

CHAIN = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
RING = CHAIN + [(6, 1)]
TRIANGLES = [(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4)]
NETS = 1000


def tables(lines):
    avg = torch.tensor(average_table(lines, 6), dtype=torch.float32)
    add = (avg > 0).float()                       # the same recipe without the division: add, do not average
    return avg, add


def answer(table, ws, card):
    """One answer for the whole drawing: run the rounds, then add up the scores of all six atoms."""
    h = card.repeat(6, 1)
    for i, w in enumerate(ws):
        h = table @ h @ w
        if i < len(ws) - 1:
            h = torch.tanh(h)
    return h.sum(0)


def largest_gap(lines_a, lines_b, kind):
    """Over 1,000 random networks: the largest difference between the answers for the two drawings."""
    g = torch.Generator().manual_seed(0)
    a, b = tables(lines_a)[kind], tables(lines_b)[kind]
    gaps = []
    for i in range(NETS):
        rounds = 1 + i % 8
        ws = [torch.randn(2, 2, generator=g) for _ in range(rounds)]
        card = torch.randn(1, 2, generator=g)
        gaps.append(float((answer(a, ws, card) - answer(b, ws, card)).abs().max()))
    return {'largest': max(gaps), 'smallest': min(gaps), 'same': sum(x < 1e-4 for x in gaps), 'nets': NETS}


def run():
    return {'chain_ring_average': largest_gap(CHAIN, RING, 0), 'chain_ring_add': largest_gap(CHAIN, RING, 1),
            'ring_triangles_add': largest_gap(RING, TRIANGLES, 1),
            'ring_triangles_average': largest_gap(RING, TRIANGLES, 0)}


if __name__ == '__main__':
    r = run()
    for name, k in [('chain of 6 and ring of 6, averaging', 'chain_ring_average'),
                    ('chain of 6 and ring of 6, adding', 'chain_ring_add'),
                    ('ring of 6 and two rings of 3, adding', 'ring_triangles_add'),
                    ('ring of 6 and two rings of 3, averaging', 'ring_triangles_average')]:
        s = r[k]
        print(f"{name}: same answer from {s['same']} of {s['nets']} random networks "
              f"(largest difference {s['largest']:.6f})")
