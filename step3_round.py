"""Step 3. One round by hand: a member's new card is the average of the own card and the friends' cards."""
import torch
from data import FRIENDS
from models import start, AVERAGE


def by_hand(cards, m):
    """The round for one member, written out: add up the own card and the friends' cards, divide by the count."""
    group = [m] + FRIENDS[m]
    total = [sum(cards[g - 1][k] for g in group) for k in range(2)]
    return {'member': m, 'group': group, 'cards': [cards[g - 1] for g in group], 'total': total,
            'new': [t / len(group) for t in total]}


def widest(cards):
    """The largest distance between the cards of any two members."""
    return float(torch.cdist(cards, cards).max())


def run():
    cards = start(0)[0].detach()
    after = AVERAGE @ cards
    c = [[round(v, 1) for v in row] for row in cards.tolist()]
    hand = [by_hand(c, 12), by_hand(c, 10)]
    for h in hand:      # the table gives the same card as the sum by hand
        assert all(abs(a - b) < 1e-5 for a, b in zip(h['new'], after[h['member'] - 1].tolist()))
    return {'hand': hand, 'before': c, 'after': after.tolist(), 'widest_before': widest(cards),
            'widest_after': widest(after)}


if __name__ == '__main__':
    r = run()
    for h in r['hand']:
        print(f"member {h['member']} and friends {h['group'][1:]}: cards {h['cards']}")
        print(f"   sum {[round(t, 2) for t in h['total']]} divided by {len(h['group'])} = new card {[round(v, 2) for v in h['new']]}")
    print(f"largest distance between two cards: {r['widest_before']:.2f} before the round, {r['widest_after']:.2f} after")
