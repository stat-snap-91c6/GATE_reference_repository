"""
GATE reference implementation (v1.0 specification reconstruction + v1.1 transferability extension)
Qian Zhao / Kimi collaborator draft, 2026-10-25.

Spec resolutions (documented, cf. paper v0.2, Sec. 4):
  S1  Seed ring: 8 nodes, ring topology.
  S2  Arrivals 9..N each attempt m=2 attachments, processed sequentially.
  S3  Target sampling: with prob p_ep=0.95 pick a uniformly random endpoint of a uniformly
      random alive edge (flow-proportional, no global info); else uniform over arrived nodes.
  S4  Capacity B=8 LEASE slots per node. A node accepts a new lease iff it currently holds
      fewer than B leases. Capacity is enforced on both endpoints of a forming edge.
      Lease order per node = arrival order of its leases (FIFO eviction of oldest).
  S5  Admission (hereditary/graded): a non-elite-born seeker may not attach to a SATURATED hub
      (leases full); it may still attach to non-saturated hubs (the open flank).
      Elite-born: first-link target held top-k degree at the seeker's birth (k=20 default).
      Graded openness: block with probability admission_q.
  S13 No self-links: a seeker that samples itself as target records a failed
      attempt (this also keeps lineage roots strictly decreasing, so the
      lineage metric always terminates).
  S6  Eviction, lease regimes: saturated node sheds its oldest lease; the victim is distressed.
  S7  Transferability (v1.1, property regimes): the first-link edge of an elite-born node is a
      FREEHOLD bequest — it is removed from the lease queue, immune to FIFO eviction and to
      pruning, and does not count against either endpoint's lease capacity B (the bequeathed
      slot sits outside the churning lease pool). All other edges remain leases under S4/S6.
      Lineage of a node = root of its first-link parent chain.
  S8  Pruning (flat regime): every 50 arrivals, the top 3% of nodes by degree have their
      NEWEST links cut until they sit at median degree + 1.
  S9  Failed birth attempt: the seeker loses that attachment attempt (no retry within the
      arrival's own two attempts).
  S10 Snapshots every 250 arrivals from t=500 for turnover; turnover = 1 - |A intersection B|/20.
  S11 No node death in v1.0/v1.1; transferability operates through bequest (S7).
  S12 Re-seek cascade: a node that loses an edge (eviction or pruning) and is distressed
      re-seeks once: it immediately attempts one new attachment under the same admission,
      capacity and eviction rules. Evictions caused by a re-seek distress the next victim,
      which is queued; a node re-seeks at most once per cascade wave. Cascades are capped at
      200 re-seek events per arriving node. (v0.1 draft text: "evicted partners may re-seek
      later".)
"""

import numpy as np
from collections import deque


class Params:
    def __init__(self, N=3000, m=2, p_ep=0.95, B=8, seed_ring=8,
                 regime="open",            # flat | open | hereditary | graded
                 transferability="lease",  # lease | property
                 admission_q=1.0,          # graded openness: prob. a non-elite is blocked at a saturated hub
                 cap_on=True, reseek_on=True,
                 prune_every=50, prune_frac=0.03,
                 snapshot_from=500, snapshot_every=250, elite_k=20,
                 cascade_cap=200,
                 rng=None):
        self.N, self.m, self.p_ep, self.B = N, m, p_ep, B
        self.seed_ring = seed_ring
        self.regime, self.transferability = regime, transferability
        self.admission_q = admission_q
        self.cap_on = cap_on
        self.reseek_on = reseek_on
        self.prune_every, self.prune_frac = prune_every, prune_frac
        self.snapshot_from, self.snapshot_every = snapshot_from, snapshot_every
        self.elite_k = elite_k
        self.cascade_cap = cascade_cap
        self.rng = rng or np.random.default_rng(0)


def run_gate(P: Params):
    """Run one GATE simulation; return dict of metrics."""
    rng = P.rng
    N = P.N
    B = P.B if P.cap_on else 10**9
    adj = [set() for _ in range(N)]              # all neighbours (leases + freeholds)
    leases = [deque() for _ in range(N)]         # lease edges, arrival order
    n_free = np.zeros(N, dtype=np.int64)         # freehold count per node
    parent = np.full(N, -1, dtype=np.int64)      # first-link target
    elite_birth = np.zeros(N, dtype=bool)
    degree = np.zeros(N, dtype=np.int64)
    edges = []                                   # (u,v) history; some dead
    alive = set()                                # packed keys of alive edges
    edge_count = 0

    def pack(u, v):
        return (u * N + v) if u < v else (v * N + u)

    for i in range(P.seed_ring):
        u, v = i, (i + 1) % P.seed_ring
        adj[u].add(v); adj[v].add(u)
        leases[u].append(v); leases[v].append(u)
        edges.append((u, v)); alive.add(pack(u, v))
        degree[u] += 1; degree[v] += 1; edge_count += 1

    def remove_edge(u, v):
        nonlocal edge_count
        adj[u].discard(v); adj[v].discard(u)
        if v in leases[u]:
            leases[u].remove(v)
        elif u in leases[v]:
            leases[v].remove(u)
        # if the edge was already converted to freehold it is not in either queue
        alive.discard(pack(u, v))
        degree[u] -= 1; degree[v] -= 1; edge_count -= 1

    def add_edge(u, v):
        nonlocal edge_count
        adj[u].add(v); adj[v].add(u)
        leases[u].append(v); leases[v].append(u)
        edges.append((u, v)); alive.add(pack(u, v))
        degree[u] += 1; degree[v] += 1; edge_count += 1

    def to_freehold(u, v):
        """Convert edge (u,v) to a freehold bequest: leave the lease queues."""
        leases[u].remove(v); leases[v].remove(u)
        n_free[u] += 1; n_free[v] += 1

    def evict_lease(u):
        """Shed the oldest lease of u. Returns the victim or -1 (no lease evictable)."""
        nonlocal edge_count
        if not leases[u]:
            return -1
        w = leases[u].popleft()
        adj[u].discard(w); adj[w].discard(u)
        leases[w].remove(u)
        alive.discard(pack(u, w))
        degree[u] -= 1; degree[w] -= 1; edge_count -= 1
        return w

    def top_k_set(arrived, k):
        if arrived <= k:
            return set(range(arrived))
        idx = np.argpartition(degree[:arrived], -k)[-k:]
        return set(idx.tolist())

    stats = {"blocked": 0, "attempts": 0, "nonelite_flank": 0, "nonelite_total": 0,
             "cascade_events": 0, "freeholds_created": 0}
    snapshots = []
    reseek_queue = deque()
    in_queue = set()

    def distress(w):
        if P.reseek_on and w not in in_queue:
            in_queue.add(w)
            reseek_queue.append(w)

    def sample_target(bound):
        """bound = number of arrived nodes; uniform fallback samples from [0, bound)."""
        if rng.random() < P.p_ep and edge_count > 0:
            while True:
                u0, v0 = edges[rng.integers(len(edges))]
                if pack(u0, v0) in alive:
                    return u0 if rng.random() < 0.5 else v0
        else:
            return int(rng.integers(bound))

    def attempt_attach(t, is_birth, elite_now, bound):
        """One attachment attempt by node t. bound = arrived-node count (target universe)."""
        stats["attempts"] += 1
        tgt = sample_target(bound)
        if tgt == t:
            return False  # S13: no self-links; a self-target draw is a failed attempt
        if tgt in adj[t]:
            return False  # parallel edges not admitted: a duplicate channel is a failed attempt
        saturated = len(leases[tgt]) >= B
        if saturated and P.regime in ("hereditary", "graded"):
            if not elite_birth[t] and rng.random() < P.admission_q:
                stats["blocked"] += 1
                return False
        if len(leases[tgt]) >= B:
            w = evict_lease(tgt)
            if w == -1:
                return False
            distress(w)
        if len(leases[t]) >= B:
            w = evict_lease(t)
            if w == -1:
                return False
            distress(w)
        add_edge(t, tgt)
        if is_birth and parent[t] == -1:
            parent[t] = tgt
            eb = tgt in elite_now
            elite_birth[t] = eb
            if eb and P.transferability == "property":
                to_freehold(t, tgt)
                stats["freeholds_created"] += 1
        if not elite_birth[t]:
            stats["nonelite_total"] += 1
            if not saturated:
                stats["nonelite_flank"] += 1
        return True

    def run_cascade(t, elite_now, arrived):
        """Drain the re-seek queue generated within this arrival. Re-seekers are not
        births: they set no parent and receive no bequest."""
        events = 0
        while reseek_queue and events < P.cascade_cap:
            w = reseek_queue.popleft()
            in_queue.discard(w)
            attempt_attach(w, is_birth=False, elite_now=elite_now, bound=arrived)
            events += 1
            stats["cascade_events"] += 1
        reseek_queue.clear()
        in_queue.clear()

    for t in range(P.seed_ring, N):
        arrived = t + 1
        elite_now = top_k_set(t, P.elite_k)
        for _ in range(P.m):
            attempt_attach(t, is_birth=True, elite_now=elite_now, bound=arrived)
        run_cascade(t, elite_now, arrived)
        if arrived >= P.snapshot_from and (arrived - P.snapshot_from) % P.snapshot_every == 0:
            snapshots.append(frozenset(top_k_set(arrived, 20)))
        if P.regime == "flat" and arrived % P.prune_every == 0:
            med = int(np.median(degree[:arrived])) + 1
            ncut = max(1, int(P.prune_frac * arrived))
            for u in np.argsort(-degree[:arrived])[:ncut]:
                while degree[u] > med and (leases[u] or n_free[u] > 0):
                    if leases[u]:
                        w = leases[u].pop()      # newest lease
                    else:                        # only freeholds left: confiscate newest freehold
                        w = adj[u].pop()
                        n_free[u] -= 1
                        adj[w].discard(u)
                        # (confiscation of a freehold never occurs in lease regimes)
                    remove_edge(u, w)
                    distress(w)
                run_cascade(t, elite_now, arrived)
        if len(edges) > 4 * max(edge_count, 1):
            edges = [(u, v) for (u, v) in edges if pack(u, v) in alive]

    final_top20 = top_k_set(N, 20)

    def gini(x):
        x = np.sort(np.asarray(x, dtype=float))
        n = len(x)
        if x.sum() == 0:
            return 0.0
        return (2 * np.arange(1, n + 1) @ x - (n + 1) * x.sum()) / (n * x.sum())

    tv = [1 - len(a & b) / 20.0 for a, b in zip(snapshots, snapshots[1:])]

    root_cache = list(range(N))

    def root(x):
        chain = []
        r = x
        while parent[r] != -1:
            chain.append(r)
            r = parent[r]
        for c in chain:
            root_cache[c] = r
        return r

    roots_in_top = {}
    for u in final_top20:
        r = root(u)
        roots_in_top[r] = roots_in_top.get(r, 0) + 1

    outsider = float(np.mean([not elite_birth[u] for u in final_top20]))
    return {
        "gini": float(gini(degree)),
        "top20_share": float(degree[list(final_top20)].sum() / max(degree.sum(), 1)),
        "turnover": float(np.mean(tv)) if tv else float("nan"),
        "outsider_ascent": outsider,
        "elite_origin": 1.0 - outsider,
        "lineage_conc": max(roots_in_top.values()) / 20.0,
        "top20_freehold_share": float(np.mean([n_free[u] / max(degree[u], 1) for u in final_top20])),
        "blocked_share": stats["blocked"] / max(stats["attempts"], 1),
        "nonelite_flank_share": stats["nonelite_flank"] / max(stats["nonelite_total"], 1),
        "cascade_per_run": stats["cascade_events"],
        "freeholds_created": stats["freeholds_created"],
    }
