"""Exercise. No network, no training, no cards. Every member holds one number.
The instructor starts at +1, the administrator at -1, everybody else at 0.
Do three averaging rounds. A member above 0 is put on the instructor's side, below 0 on the administrator's.

Guess first: how many of the other 32 members does this get right? Then run it:  python3 exercise.py
Then change ROUNDS to 1, to 10 and to 40."""
import numpy as np
from data import N, SIDE, INSTRUCTOR, ADMIN, average_table

ROUNDS = 3


def run(rounds=ROUNDS):
    x = np.zeros(N)
    x[INSTRUCTOR - 1], x[ADMIN - 1] = 1, -1
    table = average_table()
    with np.errstate(all='ignore'):      # some numpy builds on Apple chips print a false warning here
        for _ in range(rounds):
            x = table @ x
    others = [m for m in range(1, N + 1) if m not in (INSTRUCTOR, ADMIN)]
    wrong = [m for m in others if (x[m - 1] < 0) != (SIDE[m] == 1) or x[m - 1] == 0]
    return {'right': len(others) - len(wrong), 'wrong': wrong, 'gap': float(x.max() - x.min())}


if __name__ == '__main__':
    r = run()
    print(f"{ROUNDS} rounds of averaging, no training: {r['right']} of 32 right, wrong: {r['wrong']}")
    print(f"largest gap between two members' numbers: {r['gap']:.3f}")
