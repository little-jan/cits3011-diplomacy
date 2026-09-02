# Group 42 Diplomacy Agent — Strategy Document

*What the agent is, and exactly what it does on every turn.*

This document describes the agent's decision procedure. It is a companion to the assessed report:
the report argues *why* the design was chosen and presents the experiments; this document specifies
*what the agent actually does*, in enough detail to follow the code or to reproduce the behaviour.

---

## 1. What the agent is, in one paragraph

The agent is a **one-ply, joint-order optimiser**. It does not search a game tree. Every phase it
constructs a complete set of orders for all the units it controls, scores that set with a heuristic
estimate of the position it expects *after adjudication*, and improves the set by local search until
its time budget runs out. All strategic content lives in two places: the **potential field** that
tells a unit where value is, and the **combat model** that tells it whether a given order will
actually work. Everything else is machinery for searching order sets quickly and never returning an
illegal or wasteful one.

It plays a single power, has no communication channel (No-Press), and treats the other six powers as
part of a predictable environment rather than as agents to negotiate with.

---

## 2. What it computes once, at the start of a game (`new_game`)

Paid once, then reused for the whole game. Measured at well under 0.05 s.

| Structure | What it is |
|---|---|
| `map_graph_army` | Adjacency over `LAND` and `COAST` provinces — where an army can walk |
| `map_graph_navy` | Adjacency over `WATER` and `COAST` locations, keeping coastal variants (`SPA/NC`, `BUL/EC`, …) as separate nodes — where a fleet can sail |
| `dist['A']`, `dist['F']` | All-pairs shortest path lengths on each of those graphs |
| `padj`, `pdist` | A coast-free province graph and its distances, used for opponent reasoning |
| `nodes_of` | Province → the graph nodes representing it (so `SPA` maps to `SPA/NC`, `SPA/SC` for fleets) |

Two separate mobility graphs matter more than it looks. A fleet must never be pulled towards
Budapest, and an army must never be pulled towards the Mid-Atlantic. Sharing one graph makes both
mistakes.

---

## 3. What it computes every phase (the "context")

Before choosing anything, the agent builds a picture of the board.

**3.1 Ownership and occupancy.** Which power owns each supply centre; which power has a unit in each
province; where our own units are. Dislodged units (prefixed `*`) are excluded.

**3.2 The potential field.** For each unit type $t \in \{A, F\}$ and each location $\ell$:

$$\Phi_t(\ell) = \sum_{c \in \mathcal{SC} \setminus \mathcal{O}} \gamma^{\,d_t(\ell, c)}, \qquad \gamma = 0.80$$

summed over every supply centre we do not currently own, using the mobility graph for that unit
type, ignoring anything more than 12 moves away. A location scores highly when it is close to
*several* capturable centres at once, not merely close to one.

**3.3 Opponent behaviour rates.** From the order sets observed in `update_game`, for each opponent
$q$: a Beta-smoothed support rate $\hat{s}_q = (S_q + 1)/(N_q + 4)$ and move rate
$\hat{m}_q = (M_q + 1)/(N_q + 4)$, where $N_q$ counts that power's order opportunities. The prior of
0.25 is mildly cautious and decays towards zero for a power that never acts — so a passive opponent
stops attracting defensive effort within a few turns.

**3.4 Predicted enemy movement.** Each opponent unit is rolled forward one ply under the opponents'
own objective — *step onto a centre you do not own; failing that, close the distance to one* — and
the results are accumulated, weighted by $\hat{m}_q$, into `pred(p)`: the expected number of enemy
units entering province $p$ this turn. Ties are split evenly across equally attractive destinations.

**3.5 Defence and threat.** For each province:

- **Defence** (what it costs us to take it) = $1 + \hat{s}_o \cdot \min(\text{adjacent units of } o, 2)$
  if occupied by power $o$, plus $\min(0.7 \cdot \text{pred}(p),\ 1.2)$ for contention from others.
- **Threat** (what it costs us to keep it) = $\min(1.1 \cdot \text{pred}(p),\ 2.5)$.

**3.6 Occupancy reward.** What standing on a province is worth at the end of this phase:

| Province | Spring | Fall |
|---|---|---|
| Supply centre we do **not** own | 1.00 | **3.00** |
| Supply centre we **do** own | $0.60 \times \min(\text{threat}, 1.5)$ | $1.30 \times \min(\text{threat}, 1.5)$ |
| Anything else | 0 | 0 |

Two decisions are encoded here and both are deliberate. **Fall is worth three times Spring** because
supply-centre ownership is only re-evaluated after the Fall movement phase — Spring is positioning,
Fall is banking. And **a centre we already own is worth nothing to sit on unless somebody can
actually take it**; otherwise rear-area units get pinned guarding a quiet centre nobody is
approaching, and the agent stops advancing.

The value of a location is then `occupancy_reward(province) + 1.20 × Φ_type(location)`.

---

## 4. What it does in a movement phase

### 4.1 Candidate orders

For each of our units the agent takes the engine's own `get_all_possible_orders()` list and keeps:
holds, moves, convoy orders, and supports whose supported unit is one of ours. Building candidates
by *filtering the engine's legal-order list* rather than by constructing strings guarantees the agent
never emits an order the engine will mark `void`. Lists wider than 34 orders are pruned by a
single-order score.

### 4.2 Scoring a complete order set

Given a full assignment of one order per unit, the agent counts how many of our supports back each
move, how many back each hold, which convoys are ordered, and how many of our units are heading to
the same destination. Then, per unit:

**If the unit moves,** attack strength is $1 + (\text{our supports for that exact move})$, and:

$$P(\text{success}) = \sigma\big(3.0 \cdot (\text{strength} - \text{defence} - 0.5)\big)$$

The $-0.5$ offset encodes the rule that an attack must *strictly exceed* the defence. An unsupported
attack on a holding unit therefore scores 0.18, a supported one 0.82 — which is what makes the agent
seek supports rather than throw units at defended provinces. The move contributes
$P \cdot V(\text{destination}) + (1-P) \cdot (V(\text{here}) - 0.10)$.

Four situations are detected and penalised outright, each set to $P = 0$:

| Situation | Penalty | Why |
|---|---|---|
| Moving into our own unit that is not vacating | 0.60 | Guaranteed bounce |
| Two of our units ordered to the same province | 0.90 | They bounce each other |
| Two of our units trying to swap places | 0.90 | Illegal without a convoy — both bounce |
| A `VIA` move with no matching convoy order | 0.10 | The move is void |

A further **0.40 penalty applies to reversing the previous turn's move**, which prevents the agent
oscillating between two equally valued provinces.

**If the unit stays** (hold, support, or convoy) it contributes
$\sigma\big(3.0 \cdot ((1 + \text{supports for its hold}) - \text{threat})\big) \times V(\text{here})$ —
i.e. the value of the province, discounted by the chance it gets dislodged. Supports earn nothing
directly; their value arrives through the success of whatever they support. Three penalties keep
supports honest: **0.18** for a support that backs a move nobody is making, **0.18** for supporting
a unit that is moving away, and **0.15** for guarding a province with threat below 0.30 — without
that last one the agent settles into a cosy ring of units support-holding each other against nobody.

### 4.3 The search

1. **Construct.** Each unit independently takes its best single order. This is roughly greedy-agent
   behaviour and serves as a starting point.
2. **Climb, single-unit.** Sweep units in random order; for each, adopt the best replacement order if
   it strictly improves the total. Repeat until a sweep changes nothing.
3. **Climb, pairwise.** For every ordered pair of our units $(i, j)$ and every move $m$ available to
   $i$, if $j$ has a legal support for exactly that move, evaluate setting both *simultaneously*.
   This is the step that finds supported attacks. From a plan where $i$ holds and $j$ is busy
   elsewhere, the supported attack needs two changes at once, and each change alone lowers the
   score — a single-unit climb is trapped there permanently. Larger coalitions are reached
   incrementally: once $i$ attacks $X$ with one supporter, adding a second is a single-unit step.
4. **Restart.** Perturb two random units of the best plan found so far and climb again. The
   incumbent is never discarded.
5. **Stop** at 0.45 s and return the best plan found. Measured worst case across all experiments:
   **0.463 s** against a hard 1.000 s limit.

The search is *anytime* by design — it can be cut off at any moment and still returns a complete,
legal, sane order set. Given that breaching the time limit means disqualification, this property was
worth more to us than a stronger algorithm that must run to completion.

---

## 5. What it does in the other phases

**Retreat.** Each dislodged unit retreats to the location maximising
$V(\text{destination}) - 0.5 \times \text{threat}$, with a running set of already-claimed
destinations so two of our units never retreat into the same province (which would destroy both). If
no retreat is available, it disbands.

**Adjustment (builds and disbands).** The agent computes `centres − units`.

- *Builds:* it builds the unit **type** and at the **location** maximising $\Phi_t$ — the potential
  field answers "where is the frontier?" and "should this be an army or a fleet?" simultaneously,
  since $\Phi_A$ and $\Phi_F$ are computed on different graphs. A power whose remaining targets are
  inland builds armies; a power facing coastline builds fleets. No separate rule is needed.
- *Disbands:* it removes the units with the lowest positional value, i.e. those furthest from
  anything still worth taking.

---

## 6. Behaviour you should expect to see

- **Openings** look conventional — units fan out towards the nearest cluster of neutral centres,
  because that is where the potential field peaks in 1901.
- **Supported attacks appear from about 1903** without being scripted: `A GAL S A SER - BUD`,
  `A TYR S A TRI - VEN`, `A BOH S A SIL - MUN`.
- **Spring is loose, Fall is tight.** The agent will wander in Spring and consolidate onto centres in
  Fall, because the occupancy reward triples.
- **Quiet borders are left ungarrisoned.** Against passive opponents the agent commits everything
  forward, because threat collapses to zero and owned centres become worth nothing to sit on.
- **It does not convoy well.** Convoy orders are legal candidates and scored correctly, but a
  convoyed move needs two coordinated orders and the pairwise neighbourhood only generates
  (attacker, supporter) pairs, not (army, convoying fleet) pairs. England is consequently the
  weakest power in our per-power results.

---

## 7. Known weaknesses

**Multi-front pressure is invisible.** Threat is computed per province independently, so a unit
caught between two advancing powers is not valued differently from one facing a single attacker.
This shows up sharply in the per-power results — Austria (6.7 centres) and Germany (7.3) against
15.3 averaged across the other five, in Scenario 3. Traces show the agent mounting a well-supported
defence of one border while the other collapses. Aggregating threat over a province *and its
neighbourhood*, or making the defensive weight superlinear in the number of distinct attacking
powers, is the clearest available fix.

**The potential field is tuned for passive opposition.** In the mirror tournament, disabling the
potential field in favour of a nearest-centre heuristic *outperformed* the full agent by roughly 4.6
power-adjusted centres. The likely mechanism: summing $\gamma^d$ over all unowned centres pulls units
towards the largest cluster, which is optimal when nobody contests it and self-defeating when six
identical agents race to the same ground while easier local centres go unbanked. The field's
aggregation should discount each centre by predicted enemy pressure — a quantity the opponent model
already computes.

**No adversarial reasoning.** The agent predicts opponents but does not model them predicting it, so
its supports are occasionally cut in ways a best-response model would anticipate.

---

## 8. Configuration

Every component can be switched off independently, which is how the ablation experiments are run.
Defaults are the full-strength configuration used for marking.

```python
StudentAgent(
    use_diffusion      = True,   # potential field; False → nearest-centre heuristic
    use_pair_moves     = True,   # pairwise support neighbourhood in the search
    use_opponent_model = True,   # behaviour rates + one-ply move prediction
    use_season         = True,   # Spring/Fall objective re-weighting
    use_adjustment     = True,   # build/disband/retreat policy; False → random legal
    use_local_search   = True,   # False → greedy construction only
    time_budget        = 0.45,   # seconds of search per get_actions call
)
```

## 9. Rule compliance

Uses only `random`, `math`, `time`, `collections`, `networkx` and `timeout-decorator`. Writes no
files, makes no network or API calls, uses no GPU. Memory is two distance tables over ~85 nodes,
far below the 512 MB limit. All four decorated interface methods stay within the 1-second limit,
worst case measured 0.463 s. `get_actions` is additionally wrapped so that any unexpected engine
state falls back to a random legal order set rather than forfeiting the turn.
