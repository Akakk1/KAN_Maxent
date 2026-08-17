# Methodological Closure v1 — SUMMARY

**Run dir:** `methodological_closure_full_v1`
**Protocol:** `methodological_closure_v1`
**Rows:** 3898

## Stage A — additive(λ*) vs maxnet

- Additive(λ*) − maxnet@10k: mean ΔAUC = **+0.0089** [+0.0023, +0.0152] (n=225)
- Additive(λ*) − maxnet@50k: mean ΔAUC = **+0.0077** [+0.0011, +0.0140] (n=222)
- Additive(λ*) − same-basis GAM: mean ΔAUC = **-0.0002** [-0.0006, +0.0000] (n=225); Pearson r = 0.999865

### λ* distribution (additive rows)
- path counts: {'po_random_5fold': 164, 'frozen_n_po_lt_30': 61, 'skipped_n_po_lt_5': 1}
- λ* value counts: {0.0001: 22, 0.001: 19, 0.01: 91, 0.1: 41, 1.0: 53}

## Stage B — fair Deep-2 @ λ*

- **deep2_rphi** − Additive(λ*): mean ΔAUC = **-0.0048** [-0.0099, -0.0001] (n_sp=226)
  - AWT: +0.0001 [-0.0103, +0.0111] n=40
  - CAN: +0.0053 [-0.0053, +0.0196] n=20
  - NSW: -0.0097 [-0.0221, +0.0018] n=54
  - NZ: -0.0099 [-0.0205, -0.0004] n=52
  - SA: -0.0051 [-0.0198, +0.0091] n=30
  - SWI: -0.0003 [-0.0039, +0.0029] n=30

- **deep2_rx** − Additive(λ*): mean ΔAUC = **-0.0085** [-0.0137, -0.0035] (n_sp=226)
  - AWT: -0.0065 [-0.0171, +0.0039] n=40
  - CAN: -0.0041 [-0.0105, +0.0019] n=20
  - NSW: -0.0085 [-0.0238, +0.0073] n=54
  - NZ: -0.0131 [-0.0246, -0.0027] n=52
  - SA: -0.0124 [-0.0241, -0.0011] n=30
  - SWI: -0.0021 [-0.0053, +0.0009] n=30

## Stage C — fair Deep-3 @ λ*

- **deep3_rphi** − Additive(λ*): mean ΔAUC = **-0.0211** [-0.0270, -0.0155] (n_sp=226)
  - AWT: -0.0262 [-0.0428, -0.0111] n=40
  - CAN: +0.0034 [-0.0126, +0.0206] n=20
  - NSW: -0.0230 [-0.0349, -0.0119] n=54
  - NZ: -0.0350 [-0.0473, -0.0245] n=52
  - SA: -0.0193 [-0.0362, -0.0035] n=30
  - SWI: -0.0050 [-0.0092, -0.0010] n=30

- **deep3_rx** − Additive(λ*): mean ΔAUC = **-0.0390** [-0.0463, -0.0321] (n_sp=226)
  - AWT: -0.0325 [-0.0486, -0.0174] n=40
  - CAN: -0.0033 [-0.0185, +0.0119] n=20
  - NSW: -0.0357 [-0.0502, -0.0214] n=54
  - NZ: -0.0685 [-0.0829, -0.0549] n=52
  - SA: -0.0543 [-0.0758, -0.0324] n=30
  - SWI: -0.0109 [-0.0172, -0.0048] n=30

## Stage D — Standard KAN e2e (ITT after remediation)

- e2e − Additive(λ*): mean ΔAUC = **-0.0806** [-0.0935, -0.0676]
- e2e − maxnet@10k: mean ΔAUC = **-0.0717** [-0.0859, -0.0571]
- species still failed after R2: 9/226
- remediation used: {'primary': 190, 'R2': 20, 'R1': 16}

*Methodological Closure v1*