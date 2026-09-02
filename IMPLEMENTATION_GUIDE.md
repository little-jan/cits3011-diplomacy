# IMPLEMENTATION_GUIDE.md

How `agent_42.py` is put together, in the order you would write it from scratch. Read `CLAUDE.md`
first for *why*; this document is *how*.

---

## Step 0 — Understand the interface you must satisfy

`agent_baselines.Agent` defines four methods, each wrapped in a 1-second timeout:

```python
__init__(agent_name)              # construct; no game yet
new_game(game, power_name)        # a fresh deep copy of the game + which power we are
update_game(all_power_orders)     # keep our internal copy in sync; MUST call process()
get_actions() -> list[str]        # return order strings for our units this phase
```

`update_game` receives every power's orders each phase. That is the only channel through which we
observe opponents, and it is what the opponent model is built on. The body that sets orders and
calls `process()` is prescribed by the brief and must not change — but you may compute *around* it.

Three phase types come through `game.phase_type`: `M` movement, `R` retreat, `A` adjustment
(builds/disbands). All three need handling; only `M` needs the search.

---

## Step 1 — Order parsing

Everything downstream works on parsed orders, so write this first and make it total.

```python
# (kind, loc, dest, tgt, raw)
# kinds: H hold | M move | MV move-via-convoy | SH support-hold
#        SM support-move | C convoy | R retreat | D disband | B build
```

The formats you must handle, from the brief:

```
A LON H                  H    hold
F IRI - MAO              M    move; dest = w[3]
A NWY - EDI VIA          MV   convoyed move; w[4] == 'VIA'
A WAL S F LON            SH   support-hold; tgt = w[4]
F NTH S A EDI - YOR      SM   support-move; tgt = w[4], dest = w[6]
F NWG C A NWY - EDI      C    convoy; tgt = w[4], dest = w[6]
A WAL R LON              R    retreat; dest = w[3]
A LON D / A LON B        D/B  disband / build
```

Return `None` for anything unrecognised and filter it out — never crash on an unexpected string.

Also write `prov(loc)` returning `loc.split('/')[0]`. You will use it constantly.

---

## Step 2 — Map graphs and distances (`new_game`)

```python
raw  = list(game.map.loc_type.keys())     # 82 entries, includes 'spa','bul','stp','SWI'
locs = [i.upper() for i in raw]
```

Add a node to the army graph when `loc_type` is `LAND` or `COAST`; to the fleet graph when it is
`WATER` or `COAST`. Then add edges with `game.map.abuts('A', i, '-', j)` and
`game.map.abuts('F', i, '-', j)`. The double loop over 82 locations is ~13 k calls and measures
~6 ms, so there is no need to optimise it.

Then:

- `nx.all_pairs_shortest_path_length` on each graph → `self.dist['A']`, `self.dist['F']`.
- A coast-free province graph by merging both graphs under `prov()` → `self.padj`, `self.pdist`.
- `self.nodes_of[t][province] = [graph nodes]`, so `SPA` → `['SPA/NC','SPA/SC']` for fleets.
- Precompute `self.discount = [GAMMA**d for d in range(MAX_D+1)]`.

All of this is game-invariant. Compute once; never recompute per phase.

---

## Step 3 — The potential field

```python
Φ_t(ℓ) = Σ over centres c we do NOT own of  GAMMA ** d_t(ℓ, c)      # GAMMA = 0.80, d ≤ 12
```

For each unit type, for each location, sum over targets. Distance to a multi-coast province is the
**minimum** over its nodes. Cost is ~85 locations × ~20 targets × 2 graphs per phase — a few
milliseconds.

Why not nearest-centre: a position three moves from four centres beats one two moves from a single
centre, and only a discounted sum says so. Why per unit type: a fleet must never be pulled towards an
inland centre.

Keep the `use_diffusion=False` branch, which falls back to `1/(1+d_nearest)`. It is the ablation
control and it is currently *stronger* in mirror play (see CLAUDE.md §4.5).

---

## Step 4 — The opponent model

Two halves, both fed from `update_game`.

**Behaviour rates.** Snapshot `len(game.get_orderable_locations(p))` for each power *before*
`process()`, then count `SM`/`SH` and `M`/`MV` orders per power. Estimate with a Beta prior:

```python
support_rate = (supports + 1) / (slots + 4)      # prior 0.25, → 0 for a passive power
move_rate    = (moves    + 1) / (slots + 4)
```

The prior matters. Start cautious, converge to the truth. A `StaticAgent` submits *no orders at all*,
so its rates collapse within a few turns and the agent stops wasting units on defence.

**One-ply prediction.** For each opponent unit, score its legal neighbours under the *opponents'*
objective — 2.0 for stepping onto a centre that power does not own, else `1/(1+d)` to the nearest
such centre — take the argmax (splitting ties), and accumulate `move_rate / n_ties` into
`pred[province]`. Cache the per-power nearest-centre distance map; it is reused across that power's
units.

Use the opponents' objective rather than hard-coding the known baselines, so it still applies to the
Hidden Agent and to other groups' agents.

Feed the result into:

```python
defence[p] = (1 + support_rate[owner]·min(adjacent_units, 2)  if occupied by an enemy)
           + min(0.70 · pred[p], 1.20)
threat[p]  = min(1.10 · pred[p], 2.50)
```

---

## Step 5 — Location value

```python
V(ℓ) = occupancy_reward(prov(ℓ)) + GRAD_W · Φ_type(ℓ)          # GRAD_W = 1.20
```

| Province | Spring | Fall |
|---|---|---|
| Centre we do **not** own | 1.00 | 3.00 |
| Centre we **do** own | 0.60 · min(threat, 1.5) | 1.30 · min(threat, 1.5) |
| Anything else | 0 | 0 |

**The threat-conditional own-centre value is the single most important line in the file.** A constant
own-centre value pins rear units on quiet centres and, combined with any stay-discount, makes units
swap between their own centres forever. See CLAUDE.md §4.4(b).

---

## Step 6 — Scoring a complete plan

Take `plan: dict[loc → parsed order]`. First pass, tally:

- `sup_m[(target_loc, dest)]` — our supports for each specific move
- `sup_h[loc]` — our supports for each hold
- `convoys[(army_loc, dest)]`
- `dest_cnt[province]` — how many of our units head there

Second pass, per unit:

**Moving.** `strength = 1 + sup_m[(loc, dest)]`, then

```python
P = sigmoid(COMBAT_K * (strength - defence - COMBAT_EDGE))     # K = 3.0, EDGE = 0.5
contribution = P*V(dest) + (1-P)*(V(here) - BOUNCE_PEN) - penalties
```

`COMBAT_EDGE` encodes strictly-exceed. Without it, equal strengths score 0.5 and the agent throws
units at holding defenders forever.

Force `P = 0` and add a penalty for: moving into our own non-vacating unit (0.60); two of our units
to one province (0.90); attempting a swap — our unit at the destination moving to `here`, which is
illegal without a convoy (0.90); a `VIA` move with no matching convoy order (0.10). Add 0.40 if
`(dest_prov, here)` is in `self.last_moves`, which kills two-cycle oscillation.

**Staying** (hold, support, convoy):

```python
contribution = sigmoid(COMBAT_K * ((1 + sup_h[loc]) - threat[here])) * V(here)
```

Supports earn nothing directly — their value arrives through the unit they support, which is exactly
what makes the search discover them. Then penalise idle supports: 0.18 for backing a move nobody is
making, 0.18 for supporting a unit that is moving away, 0.15 for guarding a province with
`threat < 0.30`. Without that last one, units settle into a ring support-holding each other against
nobody.

---

## Step 7 — Candidate generation

For each of our units, filter `game.get_all_possible_orders()[loc]`. Keep holds, moves, convoys, and
supports **whose supported unit is one of ours**. Cap at 34 per unit, ranked by a single-order score.

Filtering the engine's list rather than building strings is what guarantees no `void` orders. This is
not a stylistic preference — treat it as a rule.

Build a support index while you are here: `sup_index[supporter][(target, dest)] = order`. The pairwise
neighbourhood depends on this being an O(1) lookup.

---

## Step 8 — The search

```
construct:  each unit takes its best single order (≈ greedy baseline)
repeat until deadline:
    climb:
        repeat up to 24 times:
            single-unit sweep   — shuffle units, adopt best strict improvement each
            pairwise sweep      — for each (i,j), for each move m of i,
                                  if j can support m, try setting both together
            break if neither improved
    if the result beats the incumbent, adopt it
    perturb 2 random units of the incumbent and climb again
```

Check `time.perf_counter() > deadline` inside **both** sweeps, not just between climbs — a single
pairwise sweep over many units is not negligible.

The pairwise sweep is the whole reason the agent works. Do not remove it as an "optimisation".

Store `self.last_moves` at the end for the next turn's reverse penalty.

---

## Step 9 — Retreat and adjustment

**Retreat.** Best destination by `V(dest) − 0.5·threat(dest)`, tracking claimed provinces so two of
our units never retreat into the same one (both would die). Disband if no retreat exists.

**Adjustment.** `delta = len(centers) − len(units)`.
- `delta > 0`: build the type and location maximising `Φ_t`. Since `Φ_A` and `Φ_F` use different
  graphs, this answers "where is the frontier" and "army or fleet" at once — no separate rule needed.
  Cap at `delta`, one per home centre.
- `delta < 0`: disband the lowest-value units.
- `delta == 0`: return `[]`.

---

## Step 10 — Safety

Wrap the whole `get_actions` body in `try/except`, falling back to random legal orders. Budget 0.45 s
of search, leaving ~2× headroom under the 1 s limit. Measure the worst case before submitting:

```python
t = time.perf_counter(); orders = agent.get_actions(); peak = max(peak, time.perf_counter() - t)
```

---

## Testing approach

Fastest useful loop, in order:

1. **Trace one game.** Instantiate the agent + 6 `StaticAgent`s manually, print units and orders each
   phase. Every pathology in CLAUDE.md §4.4 was visible within twenty phases. Do this before any
   statistical run.
2. **Single seeded game** for a quick regression signal.
3. **Full protocol** (`experiment`, all 7 powers × N reps) for headline numbers.
4. **Ablation** (`ablation`, same seed across variants) for the report.

Paired seeding across variants matters — opponent draws must be identical or the comparison is noise.
Expect ~60–100 s per Scenario-2 game on one core.
