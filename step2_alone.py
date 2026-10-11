"""Step 2. A plain network: it reads one member's card and nothing else. It is told two names."""
from statistics import mean
from models import many, train


def run():
    runs = many(rounds=0)
    right = [r['right'] for r in runs]
    return {'mean': mean(right), 'min': min(right), 'max': max(right), 'of': runs[0]['of'],
            'moved': runs[0]['moved'], 'told_right': min(r['told_right'] for r in runs), 'runs': len(runs),
            'example': train(0, rounds=0)}


if __name__ == '__main__':
    r = run()
    print(f"plain network, told 2 names, {r['runs']} different random starts")
    print(f"right on the other {r['of']}: {r['mean']:.1f} on average (from {r['min']} to {r['max']})")
    print(f"it gets the 2 told members right in every run: {r['told_right'] == 2}")
    print(f"cards that training changed: {r['moved']} of 34")
