# -*- coding: utf-8 -*-
"""batch.py — checkpointed batch runner, respawn-safe. Protocol D5.

Usage: python3 batch.py <config> <target_n> [max_per_invocation]
Each completed seed appends its row to seeds_<config>.csv IMMEDIATELY,
so a SIGKILL only loses the in-flight seed. Safe to re-run: it skips
seeds already present in the CSV. Seed set = 1..target, identical across
configs (paired design); rng = default_rng(seed*7919+13). Code state:
gate_sim.py with S13 (no self-links) — see archive_v02_data/ for the
pre-S13 D3/D4 data, superseded.
"""
import sys, time, csv, os

BASE = "/mnt/agents/output/gate_reference"
sys.path.insert(0, BASE)

FIELDS = ['gini','top20_share','turnover','outsider_ascent','elite_origin','lineage_conc',
          'top20_freehold_share','blocked_share','nonelite_flank_share','cascade_per_run',
          'freeholds_created']
COLS = ["seed"] + FIELDS

KWS = {
 'R1_flat_lease':      dict(regime='flat', transferability='lease'),
 'R2_open_lease':      dict(regime='open', transferability='lease'),
 'R3_hered_lease':     dict(regime='hereditary', transferability='lease'),
 'R4_open_property':   dict(regime='open', transferability='property'),
 'R5_hered_property':  dict(regime='hereditary', transferability='property'),
 'R0_baseline_nocap':  dict(regime='open', transferability='lease', cap_on=False),
 'SENS_noseek_flat':   dict(regime='flat', transferability='lease', reseek_on=False),
 'SENS_noseek_open':   dict(regime='open', transferability='lease', reseek_on=False),
 'G2_q0.0':  dict(regime='graded', transferability='lease', admission_q=0.0),
 'G2_q0.25': dict(regime='graded', transferability='lease', admission_q=0.25),
 'G2_q0.5':  dict(regime='graded', transferability='lease', admission_q=0.5),
 'G2_q0.75': dict(regime='graded', transferability='lease', admission_q=0.75),
 'G2_q1.0':  dict(regime='graded', transferability='lease', admission_q=1.0),
 'rob_open_k5':   dict(regime='open', transferability='lease', elite_k=5),
 'rob_open_k10':  dict(regime='open', transferability='lease', elite_k=10),
 'rob_open_k20':  dict(regime='open', transferability='lease', elite_k=20),
 'rob_open_k30':  dict(regime='open', transferability='lease', elite_k=30),
 'rob_open_k40':  dict(regime='open', transferability='lease', elite_k=40),
 'rob_open_k50':  dict(regime='open', transferability='lease', elite_k=50),
 'rob_hered_k5':  dict(regime='hereditary', transferability='lease', elite_k=5),
 'rob_hered_k10': dict(regime='hereditary', transferability='lease', elite_k=10),
 'rob_hered_k20': dict(regime='hereditary', transferability='lease', elite_k=20),
 'rob_hered_k30': dict(regime='hereditary', transferability='lease', elite_k=30),
 'rob_hered_k40': dict(regime='hereditary', transferability='lease', elite_k=40),
 'rob_hered_k50': dict(regime='hereditary', transferability='lease', elite_k=50),
 'BIG1_flat_lease':     dict(regime='flat', transferability='lease', N=10000),
 'BIG2_open_lease':     dict(regime='open', transferability='lease', N=10000),
 'BIG3_hered_lease':    dict(regime='hereditary', transferability='lease', N=10000),
 'BIG4_open_property':  dict(regime='open', transferability='property', N=10000),
 'BIG5_hered_property': dict(regime='hereditary', transferability='property', N=10000),
}

def existing_seeds(name):
    fn = f"{BASE}/seeds_{name}.csv"
    if not os.path.exists(fn):
        return set()
    with open(fn) as f:
        return {int(row.split(',')[0]) for row in f if row.strip()}

def main():
    name = sys.argv[1]
    target = int(sys.argv[2])
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 1000000
    have = existing_seeds(name)
    need = [s for s in range(1, target + 1) if s not in have]
    if not need:
        print(f"{name}: already at {len(have)} >= {target}, nothing to do")
        return
    # Shuffle before capping: spreads the loss when the sandbox CPU limiter
    # kills an invocation mid-batch (pre-S13, a divergent seed could also
    # head-of-line block the queue; S13 removed that failure mode).
    import random
    random.Random((os.getpid() * 7919 + int(time.time())) & 0xFFFFFFFF).shuffle(need)
    need = need[:cap]
    import numpy as np
    from gate_sim import Params, run_gate
    done = 0
    fn = f"{BASE}/seeds_{name}.csv"
    for s in need:
        t0 = time.time()
        P = Params(rng=np.random.default_rng(s*7919+13), **KWS[name])
        print(f"[{name} seed {s}] run start", flush=True)
        r = run_gate(P)
        t_run = time.time() - t0
        with open(fn, 'a', newline='') as f:
            csv.writer(f).writerow([s] + [r[k] for k in FIELDS])
            f.flush(); os.fsync(f.fileno())
        done += 1
        print(f"[{name} seed {s}] done in {t_run:.2f}s", flush=True)
    print(f"{name}: +{done} seeds (total {len(have)+done})")

if __name__ == '__main__':
    main()
