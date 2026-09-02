# Diplomacy Agent Project Report
**Group 42 — _[name, student number]_ · _[name, student number]_ · _[name, student number]_**

## Basic Technique

### 1. Stochastic Hill-Climbing over Joint Order Sets

The first thing we worked out is that you cannot search this game as a tree. A power with ten units
picks from roughly $10^{10}$ joint orders per phase, and that is before considering what the other
six powers do simultaneously. So instead of searching a tree, we search the *order set itself*.

The agent builds one complete assignment of orders to all our units, scores it with a heuristic
estimate of the position expected after adjudication, and then improves it by local search: sweep
the units in random order, swap each unit's order for the best alternative if that improves the
total, repeat until nothing improves, then perturb two units and climb again from the best plan so
far. It stops at 0.45 s and returns the best it found.

The reason we chose hill-climbing over anything fancier is the 1-second limit. Breaching it means
disqualification, so an algorithm that can be interrupted at any instant and still hand back a
complete, legal order set is worth more than a stronger algorithm that has to run to completion.
Measured worst-case `get_actions` time across every experiment below was **0.463 s**.

Candidates come from filtering the engine's own `get_all_possible_orders()` rather than from
building order strings ourselves, which means we never emit an order the engine marks `void`.

**Effectiveness:** Load-bearing. In the mirror tournament, disabling the search and keeping only the
greedy construction it starts from costs **−2.52 power-adjusted centres**, the worst result of any
ablation. But on its own it is not enough — a single-unit neighbourhood cannot discover supported
attacks, which is what technique 3 exists to fix.

## Novel Techniques

### 2. Discounted Supply-Centre Potential Field over Separate Mobility Graphs

The greedy baseline walks each unit towards the single nearest enemy centre. We replaced that with a
field that sums a discounted contribution from *every* centre we do not own:

$$\Phi_t(\ell) = \sum_{c \,\notin\, \text{ours}} 0.80^{\,d_t(\ell,\,c)}$$

computed separately on the army graph and the fleet graph, with all-pairs distances precomputed once
per game. Two things motivated this. First, nearest-centre is blind to clustering — a position three
moves from four centres is better than one two moves from a single centre, and only a discounted sum
says so. Second, a shared graph pulls fleets towards inland centres they can never enter.

We originally implemented this as **multi-hop value diffusion** (repeated neighbour-averaging of a
supply-centre indicator, DumbBot-style) and had to throw it away. Each averaging step divides by node
degree, so the field decays roughly as $1/5^k$ and goes numerically flat past about four hops. In a
Scenario 1 game the agent reached 7 centres and then stalled completely — with no gradient left,
every order scored the same and units shuffled between their own centres for fourteen straight
years. Swapping diffusion for the discounted-distance field fixed the stall outright: same game,
same seed, solo win in 1905.

**Effectiveness:** Genuinely mixed, and this is our most interesting result. Against the baseline
scenarios it is a clear win (7 → 18 centres on the trace above). In the mirror tournament it is
*actively harmful* — disabling it in favour of nearest-centre scored **+4.17 adjusted centres**, the
best of any variant, against the full agent's −0.42. The mechanism is that summing over all centres
pulls everyone towards the same big cluster, which is optimal when nobody contests it and
self-defeating when six identical agents race for the same ground while easy local centres go
unbanked. We validated this technique only against passive opponents and over-generalised from that.

### 3. Pairwise (Coalition) Neighbourhood in the Local Search

This is the technique that makes the agent actually work. In Diplomacy an attack on an occupied
province has strength 1, the defender has strength 1, and the attacker must *exceed* the defender —
so it fails. Only `A → X` **together with** `B S A → X` dislodges anything, and neither half improves
the position on its own. A hill-climb that changes one unit at a time is therefore permanently stuck:
both single changes lower the score, so it never finds the pair.

We added a second neighbourhood that mutates an (attacker, supporter) pair simultaneously. For every
pair of our units and every move available to the first, if the second has a legal support for
exactly that move, both are evaluated together. Supports are indexed by `(target, destination)` at
candidate-generation time so the lookup is $O(1)$, and the whole pass runs in well under a
millisecond. Three-unit coalitions come for free: once a unit is already attacking with one
supporter, adding a second is an ordinary single-unit improvement.

**Effectiveness:** Qualitatively decisive, quantitatively hard to isolate. Supported attacks appear
in the agent's own order sets from 1903 onwards without being scripted — `A GAL S A SER - BUD`,
`A TYR S A TRI - VEN`, `A BOH S A SIL - MUN`. Scenario 1 is essentially a test of this technique
alone (Static units never retreat, so a dislodged unit dies permanently and the whole scenario
reduces to reliably manufacturing 2-v-1 attacks), and we win it 100 % with zero variance. In the
mirror tournament the ablation reads +1.02 ± 1.07, which is inside noise — but that tournament pits
it against copies of itself, so both sides lose the same capability.

### 4. Predictive Opponent Modelling

Our first threat model treated any adjacent enemy unit as a uniform threat. Against Static that is
harmless, but against Greedy it is badly wrong: greedy units are strongly directional, so one
province may face two adjacent enemies and zero real pressure while its neighbour faces both.

So we model *where* opponents go, not how many are nearby. Two parts. First, from the order sets
observed every turn we keep Beta-smoothed estimates of each power's support rate and move rate,
$(S+1)/(N+4)$, with a mildly cautious prior of 0.25 that decays to zero for a power that never acts —
which is why the agent stops garrisoning against Static within a few turns. Second, we roll each
opponent unit forward one ply under the opponents' own objective (step onto a centre you do not own,
otherwise close the distance to one) and accumulate the expected arrivals per province, weighted by
that power's observed move rate. The result feeds both the estimated defence of provinces we want to
attack and the threat against provinces we hold.

We deliberately used the opponents' *objective* rather than hard-coding the known baselines, so the
technique still applies to the Hidden Agent and to Scenario 4.

**Effectiveness:** The single largest measured improvement in the project. Scenario 2 went from
12.36 ± 6.57 centres and 42.9 % wins to **14.81 ± 5.28 and 66.7 %** on the same opponent pool.
Disabling it in the mirror tournament costs −1.10 adjusted centres, second-worst after removing
search entirely.

### 5. Season-Aware Objective and Adjustment Policy

Supply-centre ownership is only re-evaluated after the Fall movement phase, so standing on a centre
in Spring is worth much less than standing on one in Fall. We weight the occupancy reward
accordingly (3.0 in Fall, 1.0 in Spring for a centre we do not own). Builds, disbands and retreats
all reuse the potential field: we build the unit type and location maximising $\Phi_t$, which answers
"where is the frontier" and "army or fleet" at the same time.

The decision that mattered most here was making the value of our *own* centres proportional to the
threat against them. Our first value model gave every owned centre a large constant value plus a 0.85
discount for staying put, and that produced two failures at once: rear units were pinned guarding
centres nobody could take, and — because moving to an equally valued centre scored 1.6 against
$0.85 \times 1.6$ for staying — units continuously **swapped between their own centres**. That was a
real local optimum of a badly specified objective, not a search bug. Tying own-centre value to threat
means a quiet centre is worth nothing to sit on and the unit is freed to advance.

Two smaller repairs belong here. We detect **head-to-head swaps** (two of our units cannot exchange
places without a convoy, so such plans score as bounces) and penalise **reversing the previous
turn's move**, which kills the two-cycle oscillations directly.

**Effectiveness:** Disabling the season weighting and the adjustment policy together costs −1.07
adjusted centres in the mirror tournament. The value-model redesign it belongs to is worth far more
than that number suggests — it is the difference between the 7-centre stall and the 18-centre win.

### 6. Combat Model (supporting all of the above)

Everything feeds one adjudication estimate. For a move with $k$ supports against estimated defence
$d$: $P = \sigma(3.0 \cdot (1 + k - d - 0.5))$. The $-0.5$ encodes that an attack must strictly
exceed the defence. Before we added that offset, equal strengths scored 0.5 and the agent cheerfully
issued hopeless unsupported attacks — `A SER - BUD` for eight consecutive years in one trace. With
it, an unsupported attack on a holding unit scores 0.18 and a supported one 0.82, which is what makes
the agent go looking for supports.

## Effectiveness Assessment

All experiments follow the marking protocol: the agent plays each power in turn against a fresh draw
of opponents, games end in 1920, scores capped at 18.

| Scenario | Avg. centres | Win rate | Defeat | n | Rubric band |
|---|---|---|---|---|---|
| 1 (vs Static) | **18.00 ± 0.00** | **100 %** | 0 % | 14 | > 90 % → 5/5 |
| 2 (Random/Attitude/Greedy) | **14.81 ± 5.28** | **66.7 %** | 4.8 % | 21 | > 50 % *and* > 13 SC → 5/5 |
| 3 (+ Hidden proxy) | **12.95 ± 6.46** | **52.4 %** | 9.5 % | 21 | > 40 % *and* > 12 SC → 5/5 |

**On Scenario 3:** the Hidden Agent is not distributed with the project, so we used the closest
calibrated stand-in we had — our own agent with the opponent model disabled, which measures 42.9 % in
Scenario 2 against the brief's stated ~50 %. Exactly one sits in each game. This proxy is
architecturally identical to us minus one component, so it fights the way we do; a genuinely
different agent could exploit weaknesses this experiment cannot surface, and 52.4 % should be read as
optimistic rather than predictive. Head-to-head in those same games the proxy managed 6.90 centres
and 9.5 % wins against our 12.95 and 52.4 %.

### Ablation Study (mirror tournament, 27 games, 190 seats)

Six variants — the full agent plus five single-component ablations — seated randomly across the seven
powers, standing in for other groups' agents of varying quality. Scores are adjusted by subtracting
each power's mean across all games, because Turkey averages 10.8 centres in mirror play and Austria
averages 1.2.

| Variant | Adjusted SC | n seats |
|---|---|---|
| − potential field (T2) | **+4.17 ± 0.72** | 36 |
| − pairwise neighbourhood (T3) | +1.02 ± 1.07 | 24 |
| **full agent** | −0.42 ± 0.66 | 39 |
| − season/adjustment (T5) | −1.07 ± 0.80 | 16 |
| − opponent model (T4) | −1.10 ± 0.42 | 32 |
| − local search (T1) | **−2.52 ± 0.32** | 42 |

Seat counts are uneven and the additive power correction is crude, so the ordering of the middle
three is not reliable. The two ends are: removing the search is clearly the worst thing you can do,
and removing the potential field clearly *helps* in mirror play.

### What Works Well

- **Pairwise support coordination:** the technique that turns a greedy chaser into something that can
  actually take defended centres. Scenario 1 is 100 % because of it.
- **Opponent modelling:** biggest single measured jump, 42.9 % → 66.7 % in Scenario 2.
- **Local search:** worst ablation result, so it is doing real work under everything else.
- **Anytime design:** 0.463 s worst case against a 1.000 s limit, with a fallback to random legal
  orders if the engine ever surprises us. No forfeits in any game we ran.

### Power-Specific Performance (Scenario 3)

| AUS | ENG | FRA | GER | ITA | RUS | TUR |
|---|---|---|---|---|---|---|
| 6.7 | 15.7 | 13.7 | **7.3** | 14.3 | 18.0 | 15.0 |

- **Russia, England, Turkey, Italy, France:** strong, averaging 15.3 centres between them.
- **Austria and Germany:** clearly our weak spot, at less than half that. These are the two powers
  with the most land neighbours and no defensible flank, and the cause is specific — our threat model
  scores each province independently, so it cannot express "this unit is caught between two fronts".
  Traces show the agent mounting a well-supported defence of one border while the other collapses.
- **England** is strong here but for a fragile reason: it does well when it is left alone. Our convoy
  handling is weak — convoy orders are scored correctly but a convoyed move needs two coordinated
  orders and our pair neighbourhood only generates (attacker, supporter) pairs, not (army, convoying
  fleet) pairs.

### Key Insights

- **The coordination barrier is the whole game.** Any method that scores units independently cannot
  dislodge a holding unit, ever. Adding a two-unit neighbourhood to an otherwise ordinary hill-climb
  was worth more than any amount of extra search depth.
- **Most of our progress came from fixing the objective, not the search.** The 7-centre stall, the
  own-centre shuffling, and the eight years of hopeless attacks on Budapest were all badly specified
  value functions that the search then optimised faithfully. Three of our biggest gains were sign
  errors and missing terms in the evaluation, not algorithmic improvements.
- **A technique validated against passive opponents may not survive competent ones.** The potential
  field is the clearest case: unambiguously good in Scenarios 1–3, actively harmful in mirror play.
  The fix we would make with more time is to discount each centre's contribution by predicted enemy
  pressure — a quantity the opponent model already computes — so the field collapses towards
  nearest-centre when the map is crowded and keeps its clustering benefit when it is not.
- **Sample sizes here are small.** A 66.7 % win rate over 21 games has a 95 % interval of roughly
  ± 20 points. The scenario figures should be read as "comfortably above the threshold, precise value
  uncertain".

## Rule Compliance

Uses only `random`, `math`, `time`, `collections`, `networkx` and `timeout-decorator`. No files
written, no network or API calls, no GPU. Memory is two distance tables over ~85 nodes, far below
512 MB. All four decorated methods stay inside the 1-second limit, worst case 0.463 s. No baseline
code was copied; `build_map_graphs` follows the construction the brief explicitly permits re-using,
and all evaluation, search and policy code is our own.

**References.** [1] D. Norman, *DumbBot algorithm description*, DAIDE — the neighbour-averaging
valuation we implemented, measured and replaced. [2] Russell & Norvig, *AIMA* 4th ed., Ch. 4 — local
search with random restarts. [3] Paquette et al., "No-Press Diplomacy: Modeling Multi-Agent
Gameplay", NeurIPS 2019 — the engine and problem setting.
