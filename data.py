"""Zachary's karate club (1977): 34 members, 78 friendships, and the side each member took in the split.

Member 1 is the instructor ("Mr. Hi"), member 34 is the administrator ("John A"). A line joins two members who
met outside the club. The numbers are the ones Zachary used. The same data ships with networkx
(nx.karate_club_graph, members numbered from 0 there); tests.py compares the two when networkx is installed."""
import numpy as np

N = 34
INSTRUCTOR, ADMIN = 1, 34
LINES = [
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 11), (1, 12), (1, 13), (1, 14),
    (1, 18), (1, 20), (1, 22), (1, 32), (2, 3), (2, 4), (2, 8), (2, 14), (2, 18), (2, 20), (2, 22), (2,
    31), (3, 4), (3, 8), (3, 9), (3, 10), (3, 14), (3, 28), (3, 29), (3, 33), (4, 8), (4, 13), (4, 14),
    (5, 7), (5, 11), (6, 7), (6, 11), (6, 17), (7, 17), (9, 31), (9, 33), (9, 34), (10, 34), (14, 34),
    (15, 33), (15, 34), (16, 33), (16, 34), (19, 33), (19, 34), (20, 34), (21, 33), (21, 34), (23, 33),
    (23, 34), (24, 26), (24, 28), (24, 30), (24, 33), (24, 34), (25, 26), (25, 28), (25, 32), (26, 32),
    (27, 30), (27, 34), (28, 34), (29, 32), (29, 34), (30, 33), (30, 34), (31, 33), (31, 34), (32, 33),
    (32, 34), (33, 34)
]
# members who went with the instructor; the other 17 stayed with the administrator
WITH_INSTRUCTOR = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 17, 18, 20, 22]
SIDE = {m: 0 if m in WITH_INSTRUCTOR else 1 for m in range(1, N + 1)}   # 0 = instructor, 1 = administrator
FRIENDS = {m: sorted([b for a, b in LINES if a == m] + [a for a, b in LINES if b == m]) for m in range(1, N + 1)}


def average_table(lines=LINES, n=N):
    """Row m holds the recipe for member m's new card: an equal share of the own card and of every friend's card."""
    t = np.eye(n)
    for a, b in lines:
        t[a - 1, b - 1] = t[b - 1, a - 1] = 1
    return t / t.sum(1, keepdims=True)
