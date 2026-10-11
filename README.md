# A graph neural network from scratch

A small graph neural network, built in seven steps on one famous data set: Zachary's karate club. The network gets the 78 friendship lines and the side of two members. It sorts the other 32 members.

The data is inside `data.py`. Nothing is downloaded.

## Run

```bash
pip install torch numpy
./run_all.sh
```

About three minutes on a laptop. Each step also runs alone, for example `python3 step3_round.py`. `python3 tests.py` runs the checks.

## Steps

| file | what it does | result on our run |
|---|---|---|
| `step1_club.py` | the club as dots and lines | 34 members, 78 lines, 17 and 17; 11 lines cross between the sides |
| `step2_alone.py` | a plain network: one card of two numbers per member, no lines | 16.2 of 32 right (50 runs); training changes 2 of 34 cards |
| `step3_round.py` | one round by hand: a member's new card is the average of the own card and the friends' cards | member 12: (−0.2, 0.9) and (−0.6, −0.6) give (−0.4, 0.15) |
| `step4_train.py` | the same network with one round | 30.8 of 32 right; training changes 31 of 34 cards; renumbering the members changes no score |
| `step5_names.py` | every pair of named members, one from each side | average 26.9; best pair the two leaders (30.8); worst pair members 9 and 25 (18.8) |
| `step6_rounds.py` | 1 to 64 rounds | 31.0 for 2 to 16 rounds; 28.0 for 32; 16.0 for 64 in every run |
| `step7_twins.py` | two drawings one network cannot tell apart | averaging: chain = ring; adding: ring of six = two triangles |
| `exercise.py` | no training at all: +1, −1 and 0, then averaging | guess first, then run it |

## The model

`models.py`, the whole forward pass:

```python
def forward(cards, tables, rounds, average=AVERAGE):
    h = cards
    for i, w in enumerate(tables):
        if i < rounds:
            h = average @ h              # every member: the average of the own card and the friends' cards
        h = h @ w                        # the same weight table for every member
        if i < len(tables) - 1:
            h = torch.tanh(h)            # squash between rounds
    return h
```

Training: Adam, rate 0.02, 200 steps, cross-entropy on the named members only.

## Limits

- One club with 34 members. Most lines (67 of 78) join members of the same side, so the task is easy.
- The cards start as random numbers. Real data gives every dot facts of its own.
- Every result is an average of 50 runs (5 runs per pair in step 5).
- Kipf and Welling divide by the square roots of both friend counts. This code divides by the member's own count.

## Sources

- W. W. Zachary, "An Information Flow Model for Conflict and Fission in Small Groups", Journal of Anthropological Research, 1977.
- T. N. Kipf, M. Welling, "Semi-Supervised Classification with Graph Convolutional Networks", arXiv 1609.02907.
- Q. Li, Z. Han, X.-M. Wu, "Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning", arXiv 1801.07606.
- K. Xu, W. Hu, J. Leskovec, S. Jegelka, "How Powerful are Graph Neural Networks?", arXiv 1810.00826.
- J. Gilmer and others, "Neural Message Passing for Quantum Chemistry", arXiv 1704.01212.

MIT licence.
