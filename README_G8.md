# G8 extension to the GATE reference implementation

`gate_g8.py` is the registered `gate_sim.py` (v1.1, protocol D5 manifest) extended by ONE
parameter for the G8 mechanicality control battery (Paper 4, §4.3a):

    Params(..., bequest="first_link")   # default; identical to registered behaviour
    bequest="random_edge"               # C1: same trigger events, freehold placed on a uniformly random alive lease edge
    bequest="universal"                 # C2: freehold granted to every node's first link regardless of parent

Two metrics added: `freeholds_per_top20`, `lease_only_overlap` (the latter is tie-sensitive
and is NOT reported in the paper). No other resolution (S1-S13) is touched.

Battery: seeds 2001-2100 (n = 100 per cell), cells R2_lease / R4_property / C1_random / C2_universal.
Raw output: `g8_battery_n100.csv`.
