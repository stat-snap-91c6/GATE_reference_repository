# GATE — Gate-Attachment Topographic Emergence

Reference implementation and registered data for the manuscript
*"Oligarchy as a topology: admission, transferability, and the freezing of
elites in a growing network"* (Paper 4, draft v0.6, 2026-10-28, Qian Zhao).

GATE is a minimal growing-network model of minority rule: a population grows
from a seed ring to N = 3,000 nodes; each arrival attempts m = 2 attachments
by endpoint sampling (p_ep = 0.95); every node holds at most B = 8 lease
slots with FIFO eviction; evicted counterparties re-seek (cascades capped at
200 per arrival). Protocol switches implement the institutional regimes:
admission gates (open / hereditary / graded), institutionalized pruning of
hubs (flat), and transferability of slots (lease / property, v1.1).

## Layout

- `code/gate_sim.py` — the registered reference implementation (v1.0 + v1.1).
  Its docstring states the thirteen specification resolutions S1–S13 that
  fix every point on which a natural-language description underdetermines
  code. S13 excludes self-links.
- `code/gate_g8.py` — the G8 mechanicality control battery of §4.3a: the
  registered code extended by a single `bequest` placement parameter
  (`first_link` default / `random_edge` / `universal`). With the default it
  reproduces `gate_sim.py` bit-for-bit; it touches no other resolution.
  Contract: `README_G8.md`.
- `code/batch.py` — checkpointed batch runner, registered protocol D5
  (paired seeds 1..target identical across all configurations;
  rng = np.random.default_rng(seed*7919 + 13)).
- `data/` — every registered results CSV the manuscript's tables and figures
  are computed from: headline regimes R0–R5 (n = 100 per cell; R0 n = 40
  exploratory), graded-admission sweep G2 (n = 30 per point), elite-set-size
  sweep rob_{open,hered}_k{5..50} (n = 30 per point), sensitivity runs
  (n = 10), scale robustness BIG1–BIG5 at N = 10,000 (n = 15), and the G8
  battery (n = 100 per cell, seed block 2001–2100).

## Reproducing the headline numbers

```bash
python code/batch.py R2_open_lease 100     # then R3_hered_lease, R4_open_property, R5_hered_property, ...
python - <<'EOF'
import pandas as pd
df = pd.read_csv("data/seeds_R4_open_property.csv", header=None,
                 names=["seed","gini","top20_share","turnover","outsider_ascent",
                        "elite_origin","lineage_conc","top20_freehold_share",
                        "blocked_share","nonelite_flank_share","cascade_per_run",
                        "freeholds_created"])
print(df["turnover"].mean(), df["gini"].mean())
EOF
```

The G8 battery (`data/g8_battery_n100.csv`) is produced by `code/gate_g8.py`
over cells R2_lease / R4_property / C1_random / C2_universal with seeds
2001–2100; see `README_G8.md`.

## Author and license

Qian Zhao (Independent Researcher), zhao.zqcm@outlook.com,
ORCID 0009-0006-6357-8688. Code and data released under the MIT License;
manuscript text is not covered.
