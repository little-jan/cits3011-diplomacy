import time
import math
import random
from collections import defaultdict, Counter

import networkx as nx
import timeout_decorator

'''
WINDOWS COMPATIBILITY NOTE:
    The timeout_decorator package may not work correctly on Windows. For local
    development on Windows, you may comment out the import and all four
    @timeout_decorator.timeout(1) lines in this file.
'''

from agent_baselines import Agent


# --------------------------------------------------------------------------- #
#  Order parsing helpers
# --------------------------------------------------------------------------- #

# A parsed order is a tuple:  (kind, loc, dest, tgt, raw)
#   kind : 'H' hold | 'M' move | 'MV' move-via-convoy | 'SH' support-hold
#          'SM' support-move | 'C' convoy | 'R' retreat | 'D' disband | 'B' build
#   loc  : location of the ordered unit          (may include a coast, e.g. SPA/NC)
#   dest : destination of the action             (or None)
#   tgt  : location of the unit being supported / convoyed (or None)

def parse_order(o):
    w = o.split()
    if len(w) < 3:
        return None
    loc = w[1]
    k = w[2]
    if k == 'H':
        return ('H', loc, None, None, o)
    if k == '-':
        if len(w) < 4:
            return None
        if len(w) > 4 and w[4] == 'VIA':
            return ('MV', loc, w[3], None, o)
        return ('M', loc, w[3], None, o)
    if k == 'S':
        if len(w) >= 7 and w[5] == '-':
            return ('SM', loc, w[6], w[4], o)
        if len(w) >= 5:
            return ('SH', loc, None, w[4], o)
        return None
    if k == 'C':
        if len(w) >= 7:
            return ('C', loc, w[6], w[4], o)
        return None
    if k == 'R' and len(w) >= 4:
        return ('R', loc, w[3], None, o)
    if k == 'D':
        return ('D', loc, None, None, o)
    if k == 'B':
        return ('B', loc, None, None, o)
    return None


def prov(loc):
    '''Strip the coast qualifier: SPA/NC -> SPA.'''
    return loc.split('/')[0]


def sigmoid(x):
    if x > 30:
        return 1.0
    if x < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# --------------------------------------------------------------------------- #
#  Student agent
# --------------------------------------------------------------------------- #

class StudentAgent(Agent):
    '''
    Group 42 agent.

    BASIC TECHNIQUE
        Stochastic hill-climbing (local search) over the joint order vector of
        every unit we control, scored by a hand-crafted evaluation of the
        expected position after adjudication.  A greedy per-unit assignment
        provides the initial candidate; the search then applies the best
        single-unit modification until no improvement is found, restarting from
        randomised perturbations while time remains.

    NEW TECHNIQUE 1 - Multi-hop supply-centre value diffusion.
    NEW TECHNIQUE 2 - Pairwise (coalition) neighbourhood in the local search.
    NEW TECHNIQUE 3 - Online opponent modelling of support / move rates.
    NEW TECHNIQUE 4 - Season-aware objective + build/disband/retreat policy.

    Each technique can be disabled independently via the constructor, which is
    how the ablation experiments in test_42.py are run.  The defaults are the
    full-strength configuration.
    '''

    # ---- tunable constants -------------------------------------------------
    TIME_BUDGET = 0.45          # seconds of search inside get_actions
    GAMMA = 0.80                # spatial discount of the potential field
    MAX_D = 12                  # distances beyond this contribute nothing
    COMBAT_K = 3.0              # sharpness of the success sigmoid
    COMBAT_EDGE = 0.5           # an attack must *exceed* the defence to succeed
    GRAD_W = 1.20               # weight of the positional gradient
    CAPTURE_F = 3.00            # occupying a centre we do not own, in Fall
    CAPTURE_S = 1.00            # ... in Spring (ownership is not yet checked)
    DEFEND_F = 1.30             # holding one of our own centres, per unit of threat
    DEFEND_S = 0.60
    REVERSE_PEN = 0.40          # cost of undoing last turn's move (anti-oscillation)
    BOUNCE_PEN = 0.10           # cost of a wasted (bounced) move
    SELF_BOUNCE_PEN = 0.90      # cost of ordering two units to the same place
    BLOCK_PEN = 0.60            # cost of moving into our own stationary unit
    IDLE_SUP_PEN = 0.18         # cost of a support that backs nothing
    QUIET_SUP_PEN = 0.15        # cost of guarding a province nobody threatens

    @timeout_decorator.timeout(1)
    def __init__(self, agent_name='Group 42',
                 use_diffusion=True,
                 use_pair_moves=True,
                 use_opponent_model=True,
                 use_season=True,
                 use_adjustment=True,
                 use_local_search=True,
                 time_budget=None):
        super().__init__(agent_name)
        self.use_diffusion = use_diffusion
        self.use_pair_moves = use_pair_moves
        self.use_opponent_model = use_opponent_model
        self.use_season = use_season
        self.use_adjustment = use_adjustment
        self.use_local_search = use_local_search
        if time_budget is not None:
            self.TIME_BUDGET = time_budget

        self.game = None
        self.power_name = None
        self.map_graph_army = None
        self.map_graph_navy = None

    # ------------------------------------------------------------------ #
    #  Set-up
    # ------------------------------------------------------------------ #

    @timeout_decorator.timeout(1)
    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name

        self._build_map_graphs()
        self._build_province_graph()

        self.scs = set(prov(s) for s in self.game.map.scs)

        # ---- opponent model state (NEW TECHNIQUE 3) --------------------
        self.obs_slots = defaultdict(int)     # order opportunities seen
        self.obs_support = defaultdict(int)   # support orders seen
        self.obs_move = defaultdict(int)      # move orders seen
        self.last_moves = set()               # (src, dst) provinces ordered last turn
        self._gain_cache = {}

    def _build_map_graphs(self):
        '''Connection graphs of the map (same construction as the baseline).'''
        self.map_graph_army = nx.Graph()
        self.map_graph_navy = nx.Graph()

        raw = list(self.game.map.loc_type.keys())
        locs = [i.upper() for i in raw]

        for r, i in zip(raw, locs):
            t = self.game.map.loc_type[r]
            if t in ('LAND', 'COAST'):
                self.map_graph_army.add_node(i)
            if t in ('WATER', 'COAST'):
                self.map_graph_navy.add_node(i)

        for i in locs:
            for j in locs:
                if i == j:
                    continue
                if self.game.map.abuts('A', i, '-', j):
                    self.map_graph_army.add_edge(i, j)
                if self.game.map.abuts('F', i, '-', j):
                    self.map_graph_navy.add_edge(i, j)

    def _build_province_graph(self):
        '''Coast-free province graph used for valuation and threat maps.'''
        adj = defaultdict(set)
        for g in (self.map_graph_army, self.map_graph_navy):
            for a, b in g.edges():
                pa, pb = prov(a), prov(b)
                if pa != pb:
                    adj[pa].add(pb)
                    adj[pb].add(pa)
        self.padj = {p: sorted(v) for p, v in adj.items()}
        self.provinces = sorted(self.padj.keys())

        pg = nx.Graph()
        for p, ns in self.padj.items():
            for n in ns:
                pg.add_edge(p, n)
        self.pdist = dict(nx.all_pairs_shortest_path_length(pg))

        # All-pairs distances on the two mobility graphs.  The map never
        # changes, so this is paid once per game.
        self.dist = {'A': dict(nx.all_pairs_shortest_path_length(self.map_graph_army)),
                     'F': dict(nx.all_pairs_shortest_path_length(self.map_graph_navy))}
        self.nodes_of = {'A': defaultdict(list), 'F': defaultdict(list)}
        for t, g in (('A', self.map_graph_army), ('F', self.map_graph_navy)):
            for n in g.nodes():
                self.nodes_of[t][prov(n)].append(n)
        self.discount = [self.GAMMA ** d for d in range(self.MAX_D + 1)]

    # ------------------------------------------------------------------ #
    #  Game state update  +  opponent modelling
    # ------------------------------------------------------------------ #

    @timeout_decorator.timeout(1)
    def update_game(self, all_power_orders):
        # Snapshot what we need *before* the engine advances the phase.
        observe = self.use_opponent_model and self.game.phase_type == 'M'
        if observe:
            slots = {p: len(self.game.get_orderable_locations(p))
                     for p in self.game.powers.keys()}

        # do not make changes to the following codes
        for power_name in all_power_orders.keys():
            self.game.set_orders(power_name, all_power_orders[power_name])
        self.game.process()

        if observe:
            for p, orders in all_power_orders.items():
                if p == self.power_name:
                    continue
                self.obs_slots[p] += slots.get(p, 0)
                for o in orders:
                    po = parse_order(o)
                    if po is None:
                        continue
                    if po[0] in ('SM', 'SH'):
                        self.obs_support[p] += 1
                    elif po[0] in ('M', 'MV'):
                        self.obs_move[p] += 1

    def _support_rate(self, p):
        '''Beta(1,3)-smoothed: prior 0.25, collapses to ~0 for passive powers.'''
        if not self.use_opponent_model:
            return 0.30
        return (self.obs_support[p] + 1.0) / (self.obs_slots[p] + 4.0)

    def _move_rate(self, p):
        if not self.use_opponent_model:
            return 0.60
        return (self.obs_move[p] + 1.0) / (self.obs_slots[p] + 4.0)

    # ------------------------------------------------------------------ #
    #  Province valuation
    # ------------------------------------------------------------------ #

    def _potentials(self, owner):
        """NEW TECHNIQUE 1: discounted supply-centre potential field.

        For every location we accumulate gamma**distance over *all* supply
        centres we do not yet own, computed separately on the army and the
        fleet mobility graph.  Unlike the baseline's nearest-centre heuristic
        this rewards positions that threaten several centres at once, it never
        flattens out at range, and it respects the fact that armies and fleets
        move on different graphs, so a fleet is never pulled towards an inland
        centre it can never reach.
        """
        targets = [p for p in self.scs if owner.get(p) != self.power_name]
        pot = {}
        for t in ('A', 'F'):
            dist = self.dist[t]
            nodes_of = self.nodes_of[t]
            tnodes = [nodes_of[c] for c in targets if c in nodes_of]
            field = {}
            for loc, dmap in dist.items():
                v = 0.0
                for group in tnodes:
                    best = self.MAX_D + 1
                    for n in group:
                        d = dmap.get(n)
                        if d is not None and d < best:
                            best = d
                    if best <= self.MAX_D:
                        if self.use_diffusion:
                            v += self.discount[best]
                        else:
                            # Ablation: depth-1 nearest-centre heuristic only
                            v = max(v, 1.0 / (1.0 + best))
                field[loc] = v
            pot[t] = field
        return pot

    # ------------------------------------------------------------------ #
    #  Situation context
    # ------------------------------------------------------------------ #

    def _nearest_gain(self, power, owner):
        """Province -> distance to the nearest centre `power` does not own."""
        cached = self._gain_cache.get(power)
        if cached is not None:
            return cached
        targets = [p for p in self.scs if owner.get(p) != power]
        out = {}
        for p in self.provinces:
            dm = self.pdist.get(p, {})
            best = 99
            for t in targets:
                d = dm.get(t)
                if d is not None and d < best:
                    best = d
                    if best == 0:
                        break
            out[p] = best
        self._gain_cache[power] = out
        return out

    def _predict_incoming(self, owner, occ):
        """Expected number of enemy units moving into each province.

        Rather than treating every adjacent enemy as an undifferentiated
        threat, we run the opponents' own objective -- get onto a centre you
        do not own, otherwise close the distance to one -- forward by one ply
        and count where they are actually likely to go.  Each prediction is
        scaled by that power's empirically observed move rate, so a power that
        has never moved contributes nothing.
        """
        pred = defaultdict(float)
        if not self.use_opponent_model:
            return pred
        self._gain_cache = {}
        for q in self.game.powers.keys():
            if q == self.power_name:
                continue
            mr = self._move_rate(q)
            if mr < 0.05:
                continue
            gain = self._nearest_gain(q, owner)
            for u in self.game.get_units(q):
                if u.startswith('*'):
                    continue
                ut, loc = u.split()[0], u.split()[1]
                g = self.map_graph_army if ut == 'A' else self.map_graph_navy
                if loc not in g:
                    continue
                best, best_s = [], -1e9
                for n in g.neighbors(loc):
                    pn = prov(n)
                    if pn in self.scs and owner.get(pn) != q:
                        sc = 2.0
                    else:
                        sc = 1.0 / (1.0 + gain.get(pn, 99))
                    if sc > best_s + 1e-9:
                        best_s, best = sc, [pn]
                    elif sc > best_s - 1e-9:
                        best.append(pn)
                if best:
                    share = mr / len(best)
                    for pn in best:
                        pred[pn] += share
        return pred

    def _context(self):
        phase = self.game.get_current_phase()
        is_fall = phase[0] == 'F' if self.use_season else True

        owner = {}
        for p in self.game.powers.keys():
            for c in self.game.get_centers(p):
                owner[prov(c)] = p

        occ = {}                                  # province -> power
        unit_set = defaultdict(set)               # power -> set of provinces
        for p in self.game.powers.keys():
            for u in self.game.get_units(p):
                if u.startswith('*'):
                    continue
                pr = prov(u.split()[1])
                occ[pr] = p
                unit_set[p].add(pr)

        our_at = {}                               # province -> our unit location
        for u in self.game.get_units(self.power_name):
            if u.startswith('*'):
                continue
            loc = u.split()[1]
            our_at[prov(loc)] = loc

        # ---- estimated combat strengths (NEW TECHNIQUE 3) --------------
        pred = self._predict_incoming(owner, occ)

        defence = {}
        threat = {}
        for p in self.provinces:
            o = occ.get(p)
            nbrs = self.padj.get(p, [])
            d = 0.0
            if o is not None and o != self.power_name:
                n_adj = sum(1 for q in nbrs if q in unit_set[o])
                d = 1.0 + self._support_rate(o) * min(n_adj, 2)
            if self.use_opponent_model:
                contest = min(0.70 * pred.get(p, 0.0), 1.20)
                t = min(pred.get(p, 0.0) * 1.10, 2.5)
            else:
                contest, t = 0.0, 0.0
                for q in nbrs:
                    oq = occ.get(q)
                    if oq is None or oq == self.power_name or oq == o:
                        continue
                    contest += 0.35 * self._move_rate(oq)
                    t += self._move_rate(oq)
                contest, t = min(contest, 1.0), min(t, 2.5)
            defence[p] = d + contest
            threat[p] = t

        # ---- occupancy reward + positional gradient --------------------
        # Owning a centre we already hold is worth nothing unless somebody can
        # take it from us, so a quiet rear centre must not pin a unit down.
        pot = self._potentials(owner)
        cap = self.CAPTURE_F if is_fall else self.CAPTURE_S
        dfw = self.DEFEND_F if is_fall else self.DEFEND_S

        occb = {}
        for p in self.provinces:
            v = 0.0
            if p in self.scs:
                if owner.get(p) != self.power_name:
                    v = cap
                else:
                    v = dfw * min(threat[p], 1.5)
            occb[p] = v

        return {'occb': occb, 'pot': pot, 'occ': occ, 'owner': owner,
                'our_at': our_at, 'defence': defence, 'threat': threat,
                'is_fall': is_fall}

    # ------------------------------------------------------------------ #
    #  Plan evaluation
    # ------------------------------------------------------------------ #

    def _score(self, plan, ctx):
        '''Heuristic value of the position expected after adjudicating `plan`.'''
        occb = ctx['occb']
        pot = ctx['pot']
        occ = ctx['occ']
        our_at = ctx['our_at']
        gw = self.GRAD_W

        sup_m = Counter()
        sup_h = Counter()
        convoys = Counter()
        dest_cnt = Counter()
        for loc, o in plan.items():
            k = o[0]
            if k == 'SM':
                sup_m[(o[3], o[2])] += 1
            elif k == 'SH':
                sup_h[o[3]] += 1
            elif k == 'C':
                convoys[(o[3], o[2])] += 1
            elif k == 'M' or k == 'MV':
                dest_cnt[prov(o[2])] += 1

        total = 0.0
        for loc, o in plan.items():
            k = o[0]
            here = prov(loc)
            utype = o[4][0]
            field = pot[utype] if utype in pot else pot['A']
            v_here = occb.get(here, 0.0) + gw * field.get(loc, 0.0)

            if k == 'M' or k == 'MV':
                dest = o[2]
                pd = prov(dest)
                v_dest = occb.get(pd, 0.0) + gw * field.get(dest, 0.0)
                pen = 0.0
                strength = 1.0 + sup_m[(loc, dest)]
                d = ctx['defence'].get(pd, 0.0)

                blocked = False
                swap = False
                if occ.get(pd) == self.power_name:
                    other = our_at.get(pd)
                    op = plan.get(other) if other is not None else None
                    if op is None or op[0] not in ('M', 'MV'):
                        blocked = True
                    elif op[0] == 'M' and k == 'M' and prov(op[2]) == here:
                        # two of our units cannot exchange places without a convoy
                        swap = True

                if (pd, here) in self.last_moves:
                    pen += self.REVERSE_PEN

                if blocked:
                    p = 0.0
                    pen += self.BLOCK_PEN
                elif swap:
                    p = 0.0
                    pen += self.SELF_BOUNCE_PEN
                elif dest_cnt[pd] > 1:
                    p = 0.0
                    pen += self.SELF_BOUNCE_PEN
                elif k == 'MV' and convoys[(loc, dest)] < 1:
                    p = 0.0
                    pen += self.BOUNCE_PEN
                else:
                    p = sigmoid(self.COMBAT_K *
                                (strength - d - self.COMBAT_EDGE))

                total += (p * v_dest
                          + (1.0 - p) * (v_here - self.BOUNCE_PEN)
                          - pen)
            else:
                # the unit stays where it is (hold / support / convoy)
                surv = sigmoid(self.COMBAT_K *
                               ((1.0 + sup_h[loc]) - ctx['threat'].get(here, 0.0)))
                total += surv * v_here

                if k == 'SM':
                    mover = plan.get(o[3])
                    if mover is None or mover[0] not in ('M', 'MV') or mover[2] != o[2]:
                        total -= self.IDLE_SUP_PEN
                elif k == 'SH':
                    held = plan.get(o[3])
                    if held is None or held[0] in ('M', 'MV'):
                        total -= self.IDLE_SUP_PEN
                    elif ctx['threat'].get(prov(o[3]), 0.0) < 0.30:
                        total -= self.QUIET_SUP_PEN
                elif k == 'C':
                    mover = plan.get(o[3])
                    if mover is None or mover[0] != 'MV' or mover[2] != o[2]:
                        total -= self.IDLE_SUP_PEN

        return total

    # ------------------------------------------------------------------ #
    #  Candidate generation
    # ------------------------------------------------------------------ #

    def _candidates(self, possible, our_locs, ctx):
        cands = {}
        our_set = set(our_locs)
        for loc in our_locs:
            keep = []
            for o in possible.get(loc, []):
                po = parse_order(o)
                if po is None:
                    continue
                k = po[0]
                if k in ('H', 'M', 'MV', 'C'):
                    keep.append(po)
                elif k in ('SM', 'SH') and po[3] in our_set:
                    keep.append(po)
            if not keep:
                keep = [('H', loc, None, None, '%s H' % loc)]
            if len(keep) > 34:
                def quick(po, _loc=loc):
                    ut = po[4][0]
                    f = ctx['pot'].get(ut, ctx['pot']['A'])
                    if po[0] in ('M', 'MV'):
                        return (ctx['occb'].get(prov(po[2]), 0.0)
                                + self.GRAD_W * f.get(po[2], 0.0))
                    return (ctx['occb'].get(prov(_loc), 0.0)
                            + self.GRAD_W * f.get(_loc, 0.0)) * 0.9
                keep.sort(key=quick, reverse=True)
                keep = keep[:34]
            cands[loc] = keep
        return cands

    # ------------------------------------------------------------------ #
    #  Movement-phase policy
    # ------------------------------------------------------------------ #

    def _movement_orders(self, deadline):
        ctx = self._context()
        possible = self.game.get_all_possible_orders()
        our_locs = [l for l in self.game.get_orderable_locations(self.power_name)
                    if possible.get(l)]
        if not our_locs:
            return []

        cands = self._candidates(possible, our_locs, ctx)

        # ---- greedy initial assignment --------------------------------
        plan = {}
        for loc in our_locs:
            best, best_s = None, -1e9
            for po in cands[loc]:
                s = self._score({loc: po}, ctx)
                if s > best_s:
                    best_s, best = s, po
            plan[loc] = best

        if not self.use_local_search:
            self.last_moves = set()
            return [plan[l][4] for l in our_locs]

        # supporter -> {(target_loc, dest): support order}
        sup_index = {}
        for loc in our_locs:
            m = {}
            for po in cands[loc]:
                if po[0] == 'SM':
                    m[(po[3], po[2])] = po
            sup_index[loc] = m

        best_plan = dict(plan)
        best_score = self._score(best_plan, ctx)
        cur = dict(best_plan)

        while time.perf_counter() < deadline:
            improved = self._climb(cur, cands, sup_index, ctx, deadline, our_locs)
            cur_score = self._score(cur, ctx)
            if cur_score > best_score:
                best_score, best_plan = cur_score, dict(cur)
            if improved and time.perf_counter() < deadline:
                continue
            # random restart around the incumbent
            cur = dict(best_plan)
            for loc in random.sample(our_locs, min(2, len(our_locs))):
                cur[loc] = random.choice(cands[loc])

        chosen = [best_plan[l] for l in our_locs]
        self.last_moves = set((prov(o[1]), prov(o[2])) for o in chosen
                              if o[0] in ('M', 'MV'))
        return [o[4] for o in chosen]

    def _climb(self, plan, cands, sup_index, ctx, deadline, our_locs):
        '''Climb to a local optimum. Returns True if anything improved.'''
        any_improved = False
        base = self._score(plan, ctx)

        for _ in range(24):
            improved = False

            # --- single-unit neighbourhood (BASIC TECHNIQUE) ------------
            order = list(our_locs)
            random.shuffle(order)
            for loc in order:
                if time.perf_counter() > deadline:
                    return any_improved
                old = plan[loc]
                best_po, best_s = old, base
                for po in cands[loc]:
                    if po is old:
                        continue
                    plan[loc] = po
                    s = self._score(plan, ctx)
                    if s > best_s + 1e-9:
                        best_s, best_po = s, po
                plan[loc] = best_po
                if best_po is not old:
                    base, improved, any_improved = best_s, True, True

            # --- pairwise support neighbourhood (NEW TECHNIQUE 2) -------
            if self.use_pair_moves:
                for i in our_locs:
                    if time.perf_counter() > deadline:
                        return any_improved
                    for j in our_locs:
                        if i == j or not sup_index[j]:
                            continue
                        oi, oj = plan[i], plan[j]
                        best_pair, best_s = None, base
                        for po in cands[i]:
                            if po[0] != 'M':
                                continue
                            sp = sup_index[j].get((i, po[2]))
                            if sp is None:
                                continue
                            plan[i], plan[j] = po, sp
                            s = self._score(plan, ctx)
                            if s > best_s + 1e-9:
                                best_s, best_pair = s, (po, sp)
                        if best_pair is None:
                            plan[i], plan[j] = oi, oj
                        else:
                            plan[i], plan[j] = best_pair
                            base, improved, any_improved = best_s, True, True

            if not improved:
                break
        return any_improved

    # ------------------------------------------------------------------ #
    #  Retreat and adjustment phases (NEW TECHNIQUE 4)
    # ------------------------------------------------------------------ #

    def _retreat_orders(self):
        possible = self.game.get_all_possible_orders()
        locs = self.game.get_orderable_locations(self.power_name)
        if not self.use_adjustment:
            return [random.choice(possible[l]) for l in locs if possible.get(l)]

        ctx = self._context()
        taken = set()
        orders = []
        for loc in locs:
            opts = [parse_order(o) for o in possible.get(loc, [])]
            opts = [o for o in opts if o is not None]
            moves = [o for o in opts if o[0] == 'R' and prov(o[2]) not in taken]
            if moves:
                def rval(o):
                    f = ctx['pot'].get(o[4][0], ctx['pot']['A'])
                    return (ctx['occb'].get(prov(o[2]), 0.0)
                            + self.GRAD_W * f.get(o[2], 0.0)
                            - 0.5 * ctx['threat'].get(prov(o[2]), 0.0))
                best = max(moves, key=rval)
                taken.add(prov(best[2]))
                orders.append(best[4])
            else:
                dis = [o for o in opts if o[0] == 'D']
                if dis:
                    orders.append(dis[0][4])
        return orders

    def _adjustment_orders(self):
        possible = self.game.get_all_possible_orders()
        locs = self.game.get_orderable_locations(self.power_name)
        if not self.use_adjustment:
            return [random.choice(possible[l]) for l in locs if possible.get(l)]

        ctx = self._context()
        n_centers = len(self.game.get_centers(self.power_name))
        n_units = len([u for u in self.game.get_units(self.power_name)
                       if not u.startswith('*')])
        delta = n_centers - n_units
        if delta == 0:
            return []

        if delta > 0:
            builds = []
            for loc in locs:
                for o in possible.get(loc, []):
                    po = parse_order(o)
                    if po is not None and po[0] == 'B':
                        builds.append(po)
            if not builds:
                return []

            def frontier(po):
                # Build where, and of the type that, sees the most reachable
                # unowned centres: the potential field already answers both.
                ut = po[4][0]
                f = ctx['pot'].get(ut, ctx['pot']['A'])
                return f.get(po[1], 0.0) + 0.001 * random.random()

            builds.sort(key=frontier, reverse=True)
            chosen, used = [], set()
            for po in builds:
                if len(chosen) >= delta:
                    break
                if prov(po[1]) in used:
                    continue
                used.add(prov(po[1]))
                chosen.append(po[4])
            return chosen

        disbands = []
        for loc in locs:
            for o in possible.get(loc, []):
                po = parse_order(o)
                if po is not None and po[0] == 'D':
                    disbands.append(po)
        disbands.sort(key=lambda po: (ctx['pot'].get(po[4][0], ctx['pot']['A'])
                                      .get(po[1], 0.0)
                                      + ctx['occb'].get(prov(po[1]), 0.0)))
        return [po[4] for po in disbands[:-delta]]

    # ------------------------------------------------------------------ #
    #  Interface
    # ------------------------------------------------------------------ #

    @timeout_decorator.timeout(1)
    def get_actions(self):
        start = time.perf_counter()
        try:
            pt = self.game.phase_type
            if pt == 'R':
                return self._retreat_orders()
            if pt == 'A':
                return self._adjustment_orders()
            return self._movement_orders(start + self.TIME_BUDGET)
        except Exception:
            try:
                possible = self.game.get_all_possible_orders()
                locs = self.game.get_orderable_locations(self.power_name)
                return [random.choice(possible[l]) for l in locs if possible.get(l)]
            except Exception:
                return []
