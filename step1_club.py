"""Step 1. The data: who is in the club, who is friends with whom, who took which side."""
from data import N, LINES, SIDE, FRIENDS, INSTRUCTOR, ADMIN


def run():
    count = {m: len(f) for m, f in FRIENDS.items()}
    a, b = set(FRIENDS[INSTRUCTOR]), set(FRIENDS[ADMIN])
    rest = set(range(1, N + 1)) - {INSTRUCTOR, ADMIN}
    return {
        'members': N, 'lines': len(LINES),
        'instructor_side': sum(s == 0 for s in SIDE.values()), 'admin_side': sum(s == 1 for s in SIDE.values()),
        'friends': count, 'friends_instructor': count[INSTRUCTOR], 'friends_admin': count[ADMIN],
        'fewest': min(count.values()), 'one_friend': [m for m in count if count[m] == 1],
        'average_friends': 2 * len(LINES) / N,
        'across': sum(SIDE[x] != SIDE[y] for x, y in LINES),          # lines that join the two sides
        'both': sorted(a & b), 'neither': sorted(rest - a - b),       # friends of both leaders, of no leader
        'lines_list': LINES, 'side': [SIDE[m] for m in range(1, N + 1)],
    }


if __name__ == '__main__':
    r = run()
    print(f"{r['members']} members, {r['lines']} lines between them")
    print(f"{r['instructor_side']} went with the instructor, {r['admin_side']} stayed with the administrator")
    print(f"the instructor has {r['friends_instructor']} friends in the club, the administrator {r['friends_admin']}")
    print(f"fewest friends: {r['fewest']} (member {r['one_friend']}); on average {r['average_friends']:.1f}")
    print(f"{r['across']} of the {r['lines']} lines join members who ended on different sides")
    print(f"friends of both leaders: {r['both']}; friends of neither: {r['neither']}")
