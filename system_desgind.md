# SYSTEM_DESIGN.md

Architecture and computation flow of `agent_42.py`. Where `IMPLEMENTATION_GUIDE.md` explains *how
to write* each piece and `CLAUDE.md` explains *why* each design decision was made, this document
answers a narrower question: **for a single call to `get_actions()`, what runs, in what order, and
what data does it pass to what.**

---

## 1. Design goals, and how they shaped the architecture

| Goal | Consequence |
|---|---|
| Must never exceed 1 s per call | Search must be **anytime** — interruptible at any point and still return a legal, complete answer |
| Joint order space is too large to search as a tree | Search the **order set directly**, not a game tree; one evaluation function scores a whole plan |
| Combat requires *coordinated* action (§ "why not per-unit scoring") | Evaluation must see the **whole plan at once**, not score units independently |
| Map-invariant computation shouldn't repeat every turn | Split state into **once-per-game** (graphs, distances) vs. **once-per-phase** (context) vs. **once-per-candidate-evaluation** (score) |
| Engine must never receive an illegal order | Candidates are always **filtered from the engine's own legal-order list**, never hand-built |

These five constraints, not any single clever idea, are what produced the three-tier state model in
Section 2 and the four-stage pipeline in Section 4.

---

## 2. State model — what's stored, and for how long

```mermaid
flowchart LR
    subgraph T0["Once per GAME — new_game()"]
        direction TB
        g1["map_graph_army, map_graph_navy<br/>(NetworkX graphs)"]
        g2["padj, pdist<br/>(coast-free province graph)"]
        g3["dist['A'], dist['F']<br/>(all-pairs shortest paths)"]
        g4["nodes_of['A'/'F']<br/>(province → graph nodes)"]
        g5["discount[]<br/>(precomputed γ^d table)"]
        g6["scs<br/>(set of all supply-centre provinces)"]
    end

    subgraph T1["Persists across PHASES — updated in update_game()"]
        direction TB
        p1["obs_slots, obs_support, obs_move<br/>(per-opponent counters)"]
        p2["last_moves<br/>(this turn's (src,dst) set, for oscillation check)"]
    end

    subgraph T2["Rebuilt every MOVEMENT phase — _context()"]
        direction TB
        c1["owner, occ, unit_set, our_at"]
        c2["pot = _potentials()"]
        c3["defence, threat<br/>(uses _predict_incoming)"]
        c4["occb<br/>(occupancy reward table)"]
    end

    subgraph T3["Exists only during ONE get_actions() call"]
        direction TB
        s1["cands<br/>(per-unit legal-order shortlist)"]
        s2["sup_index<br/>(supporter → matching order lookup)"]
        s3["plan<br/>(the candidate order-set being scored/climbed)"]
    end

    T0 -.->|read by| T2
    T1 -.->|read by| T2
    T2 -.->|read by| T3
```

**Why this split matters:** T0 costs ~6 ms and pays for itself over ~30 turns. T1 is cheap to update
(a few counter increments) but expensive to lose — it's the entire memory the opponent model has. T2
is rebuilt fresh every movement phase because ownership, occupancy and threat genuinely change every
turn. T3 is scratch space, discarded the instant `get_actions()` returns.

---

## 3. Component responsibilities

```mermaid
classDiagram
    class StudentAgent {
        +new_game(game, power_name)
        +update_game(all_power_orders)
        +get_actions() list~str~
        -_build_map_graphs()
        -_build_province_graph()
    }

    class OpponentModel {
        -obs_slots, obs_support, obs_move
        +_support_rate(power) float
        +_move_rate(power) float
        -_nearest_gain(power, owner) dict
        -_predict_incoming(owner, occ) dict
    }

    class Valuation {
        -_potentials(owner) dict
        -_context() dict
    }

    class Scoring {
        -_score(plan, ctx) float
    }

    class Search {
        -_candidates(possible, locs, ctx) dict
        -_movement_orders(deadline) list~str~
        -_climb(plan, cands, sup_index, ctx, deadline, locs) bool
    }

    class PhasePolicies {
        -_retreat_orders() list~str~
        -_adjustment_orders() list~str~
    }

    StudentAgent --> OpponentModel : update_game feeds it
    StudentAgent --> Valuation : get_actions calls _context
    Valuation --> OpponentModel : defence/threat need support_rate, pred
    StudentAgent --> Search : movement phase
    Search --> Scoring : every candidate plan is scored
    Search --> Valuation : candidates ranked by ctx values
    StudentAgent --> PhasePolicies : retreat / adjustment phases
    PhasePolicies --> Valuation : reuse the same _context()
```

Nothing here is a separate class in the actual code — `agent_42.py` is one flat `StudentAgent` class,
listed this way only to show *conceptual* boundaries. The key dependency to notice: **Scoring depends
on Valuation, Valuation depends on OpponentModel, and Search depends on both** — data flows in one
direction, nothing upstream ever reads from something computed downstream.

---

## 4. The full computation pipeline, one `get_actions()` call

This is the answer to "how is everything calculated" — every arrow below is a real function call or
data dependency in the code, in execution order.

```mermaid
flowchart TD
    START(["get_actions() called<br/>by the game engine"]) --> PT{"game.phase_type"}

    PT -->|"'R' retreat"| RET["_retreat_orders()"]
    PT -->|"'A' adjustment"| ADJ["_adjustment_orders()"]
    PT -->|"'M' movement"| CTX

    subgraph CTX_BLOCK["_context() — rebuilt fresh this phase"]
        CTX["scan game.powers:<br/>owner{}, occ{}, unit_set{}, our_at{}"]
        CTX --> POT["_potentials(owner)<br/>Φ_A(ℓ), Φ_F(ℓ) for every location"]
        POT -->|"uses T0: dist['A'], dist['F']"| POT
        CTX --> PRED["_predict_incoming(owner, occ)"]
        PRED --> RATE["_support_rate(q), _move_rate(q)<br/>per opponent q"]
        RATE -->|"reads T1: obs_slots/support/move"| RATE
        PRED --> GAIN["_nearest_gain(q, owner)<br/>cached per power per phase"]
        GAIN --> ROLL["roll each opponent unit forward<br/>one ply under their own objective"]
        ROLL --> PREDOUT["pred[province] = Σ move_rate × share"]
        PREDOUT --> DEFTH["defence[p], threat[p]<br/>per province"]
        POT --> OCCB["occb[p]<br/>Fall/Spring × owned/unowned"]
        DEFTH --> OCCB
    end

    CTX_BLOCK --> CANDS["_candidates(possible_orders, our_locs, ctx)<br/>filter engine's legal-order list<br/>build sup_index{} for O(1) support lookup"]

    CANDS --> CONSTRUCT["greedy construction:<br/>each unit alone → best single order<br/>(calls _score on 1-unit plans)"]

    CONSTRUCT --> CLIMB_LOOP{"time left<br/>before 0.45s deadline?"}

    CLIMB_LOOP -->|yes| CLIMB
    subgraph CLIMB_BLOCK["_climb() — up to 24 passes"]
        CLIMB["single-unit sweep:<br/>shuffle units, try every candidate<br/>order for each, keep if _score improves"]
        CLIMB --> PAIR["pairwise sweep:<br/>for each (i,j), for each move of i,<br/>look up matching support in sup_index[j],<br/>try both together, keep if _score improves"]
        PAIR --> IMPROVED{"anything<br/>improved?"}
        IMPROVED -->|yes| CLIMB
    end
    IMPROVED -->|no| KEEPBEST["if this climb's plan<br/>beats incumbent, adopt it"]
    KEEPBEST --> PERTURB["perturb 2 random units<br/>of the incumbent"]
    PERTURB --> CLIMB_LOOP

    CLIMB_LOOP -->|no, deadline hit| RETURN_M["record last_moves<br/>return best_plan as order strings"]

    RET --> ORDERS(["list of order strings<br/>returned to engine"])
    ADJ --> ORDERS
    RETURN_M --> ORDERS
```

### What `_score(plan, ctx)` does with the T2/T3 data, per candidate plan evaluated above

Every single `_score` call inside construction, the single-unit sweep, and the pairwise sweep runs
this same sequence — it's the innermost, most-called piece of the whole system, so its cost dominates
the time budget:

```mermaid
flowchart LR
    P["plan: dict[loc → parsed order]"] --> TALLY["one pass: count<br/>sup_m, sup_h, convoys, dest_cnt"]
    TALLY --> PERUNIT["second pass: per unit"]
    PERUNIT --> MOVE{"order kind ==<br/>M or MV?"}
    MOVE -->|yes| STRENGTH["strength = 1 + sup_m[this move]"]
    STRENGTH --> SIGMOID["P_success = σ(K·(strength − defence[dest] − 0.5))"]
    SIGMOID --> PENCHECK["apply penalties:<br/>block / self-bounce / swap /<br/>unconvoyed-VIA / reverse-move"]
    PENCHECK --> CONTRIB1["contribution =<br/>P·V(dest) + (1−P)·(V(here)−bounce_pen) − penalties"]
    MOVE -->|no, holds/supports/convoys| SURV["P_survive = σ(K·(1+sup_h[here] − threat[here]))"]
    SURV --> IDLE["idle-support / idle-convoy /<br/>quiet-guard penalty check"]
    IDLE --> CONTRIB2["contribution = P_survive · V(here) − idle_pen"]
    CONTRIB1 --> SUM["total += contribution"]
    CONTRIB2 --> SUM
    SUM --> OUT["return total<br/>(one float, the whole plan's score)"]
```

---

## 5. Sequence across a full game turn (engine ⇄ agent)

This shows how `get_actions()` fits into the larger loop the brief's `game.py` drives, and clarifies
*when* the opponent model actually learns something.

```mermaid
sequenceDiagram
    participant Engine as diplomacy engine
    participant Agent as StudentAgent
    participant State as T1 state<br/>(obs_slots/support/move)

    Engine->>Agent: get_actions()
    activate Agent
    Agent->>Agent: _context() [rebuild T2 fresh]
    Agent->>Agent: candidates → construct → climb (0.45s budget)
    Agent-->>Engine: list[order strings]
    deactivate Agent

    Engine->>Engine: collect ALL powers' orders

    Engine->>Agent: update_game(all_power_orders)
    activate Agent
    Agent->>Agent: snapshot slots BEFORE process()
    Agent->>Engine: game.set_orders(...) for every power
    Agent->>Engine: game.process() — engine adjudicates
    Engine-->>Agent: (game state now advanced)
    Agent->>State: += observed supports/moves for each opponent
    deactivate Agent

    Note over Engine,Agent: Next phase begins.<br/>T1 (opponent counters) persisted.<br/>T2 (context) will be rebuilt from scratch.
```

**Key detail worth flagging:** `update_game` counts opponent behaviour using the **slot count taken
before `process()` runs**, but the **orders dict that was already decided before adjudication**. The
opponent model therefore learns from *intent* (what they ordered), not *outcome* (what actually
succeeded) — it never has to guess whether an order was a bluff or a bounce, because it's reading the
order itself, which the engine hands to every power's `update_game` regardless of adjudication
result.

---

## 6. Cost accounting — why the budget holds

| Stage | Frequency | Measured / estimated cost | Why |
|---|---|---|---|
| `_build_map_graphs`, `_build_province_graph` | once per game | ~6 ms | 82 locations, double loop, one-time |
| all-pairs shortest paths (×3 graphs) | once per game | a few ms | NetworkX, small graph |
| `_context()` | once per movement phase | small ms | O(provinces × units), no search inside it |
| `_potentials()` | inside `_context()` | small ms | O(locations × unowned centres × 2 graphs) |
| `_predict_incoming()` | inside `_context()` | small ms | O(opponent units × their degree), cached per-power distance maps |
| `_score()` | called **hundreds of times** per phase | the dominant cost | O(units) per call — this is why it's kept to two flat passes over `plan`, no nested loops over the whole board |
| single-unit sweep | up to 24× per climb | O(units × candidates_per_unit) `_score` calls | bounded by the 34-candidate cap per unit |
| pairwise sweep | up to 24× per climb | O(units² × moves_per_unit) `_score` calls | bounded by `sup_index` giving O(1) lookup instead of O(candidates) |
| **total `get_actions()`** | every movement phase | **measured worst case 0.463 s** | budget is 0.45 s of search + candidate/context overhead; hard limit is 1.0 s |

The two expensive things — `_score` and the pairwise sweep — are both bounded by **our own unit
count**, not the board size or opponent count. This is the direct payoff of the "search the order set,
not a game tree" decision from Section 1: cost scales with $n$ (our units, typically ≤ 15), never with
the opponents' combined branching factor.

---

## 7. Retreat and adjustment phases — same valuation, simpler search

Both reuse `_context()` (so they see the same `pot`, `occb`, `threat` as movement does) but skip the
climb entirely — no coordination is possible in these phase types (each unit's retreat/build/disband
is independent of the others), so there's nothing for the pairwise neighbourhood to find.

```mermaid
flowchart LR
    subgraph Retreat
        R1["for each dislodged unit"] --> R2["score every legal retreat:<br/>V(dest) − 0.5×threat(dest)"]
        R2 --> R3["pick best UNCLAIMED destination<br/>(track a 'taken' set)"]
        R3 --> R4["no legal retreat? → disband"]
    end
    subgraph Adjustment
        A1["delta = centres − units"] --> A2{"delta sign?"}
        A2 -->|">0, build"| A3["for each home centre:<br/>score by Φ_type(location)<br/>(army graph vs fleet graph<br/>picks the unit type too)"]
        A2 -->|"<0, disband"| A4["sort our units by<br/>Φ_type(location)+occb<br/>disband the lowest -delta"]
        A2 -->|"==0"| A5["return []"]
    end
```

---

## 8. Failure containment

```mermaid
flowchart TD
    CALL["get_actions() invoked"] --> TRY{"try:"}
    TRY -->|normal path| DISPATCH["dispatch by phase_type,<br/>run pipeline as above"]
    TRY -->|any exception| CATCH["except: fall back to<br/>random legal order per unit"]
    CATCH --> TRY2{"even that fails?"}
    TRY2 -->|yes| EMPTY["return []"]
    TRY2 -->|no| CATCHOK["return random legal orders"]
    DISPATCH --> RETURN["return computed orders"]
    RETURN --> DECORATOR["@timeout_decorator.timeout(1)<br/>hard-kills the call if it overruns"]
    CATCHOK --> DECORATOR
    EMPTY --> DECORATOR
```

The `try/except` is a second, independent line of defence *inside* the 1-second budget — it exists
so that an unexpected engine state (a phase-type this code didn't anticipate, a malformed order
string from `get_all_possible_orders`) degrades to "play something legal" rather than propagating an
exception that would forfeit the turn. The `@timeout_decorator` is the brief's own hard stop; the
internal 0.45 s deadline exists so the agent *chooses* to stop well before that hard stop is ever
tested.

---

## 9. Summary — the one-sentence version of every section above

1. **State** is tiered by how often it changes: graphs once per game, opponent counters persist and
   accumulate, context rebuilt every phase, search scratch discarded every call.
2. **Computation flows one direction**: opponent model → valuation → scoring → search. Nothing
   downstream is read by anything upstream.
3. **`_score()` is the hot path**, called hundreds of times per phase, and its cost is bounded by our
   own unit count — never by the opponents' combined branching factor.
4. **Retreat and adjustment are cheap** because they need no coordination search — the same valuation
   feeds a simple per-unit best-pick instead of a climb.
5. **Two independent safety nets** — an internal try/except and the external hard timeout — guarantee
   the agent never forfeits a turn, regardless of what goes wrong upstream.