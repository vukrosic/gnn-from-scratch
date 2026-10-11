"""Nine checks of the building blocks. Run before the steps: python3 tests.py"""
import numpy as np
from data import N, LINES, SIDE, FRIENDS, average_table
from models import train
from step3_round import run as round_by_hand
from step4_train import renumbered
from step6_rounds import blur
from step7_twins import RING, TRIANGLES, CHAIN


def check(name, ok):
    print(('ok   ' if ok else 'FAIL ') + name)
    assert ok, name


def friends_per_atom(lines):
    return sorted(sum(m in ln for ln in lines) for m in range(1, 7))


if __name__ == '__main__':
    check('34 members and 78 lines; no line twice, no member joined to itself',
          N == 34 and len(LINES) == 78 == len({frozenset(ln) for ln in LINES}) and all(a != b for a, b in LINES))
    check('17 members on each side', sorted(SIDE.values()) == [0] * 17 + [1] * 17)
    try:
        import networkx as nx
        g = nx.karate_club_graph()
        same = {frozenset((a + 1, b + 1)) for a, b in g.edges} == {frozenset(ln) for ln in LINES} and all(
            (g.nodes[m - 1]['club'] == 'Mr. Hi') == (SIDE[m] == 0) for m in SIDE)
        check('the lines and sides are the ones that ship with networkx', same)
    except ImportError:
        print('skip networkx is not installed, so the data is not compared with it')
    t = average_table()
    check('every row of the averaging table adds up to 1', bool(np.allclose(t.sum(1), 1)))
    check('member 12 has one friend, so its row holds one half for itself and one half for member 1',
          FRIENDS[12] == [1] and t[11, 11] == 0.5 == t[11, 0])
    round_by_hand()
    check('the round done by hand gives the same cards as the table', True)
    check('the plain network changes 2 cards, the network with one round changes 31',
          train(0, rounds=0, steps=20)['moved'] == 2 and train(0, rounds=1, steps=20)['moved'] == 31)
    check('renumbering the members changes no score by more than 0.00001', renumbered() < 1e-5)
    check('after 64 rounds of averaging alone, no two cards are further apart than 0.001', blur()[64] < 1e-3)
    check('ring of six and two rings of three: every atom has 2 friends; the chain has two atoms with 1',
          friends_per_atom(RING) == friends_per_atom(TRIANGLES) == [2] * 6 and friends_per_atom(CHAIN)[:2] == [1, 1])
    print('all checks passed')
