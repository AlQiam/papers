# Plan: gray-box adversarial search against DOTE

Implementing Namyar et al., HotNets '24, Sections 3.2 and 4, on top of the DOTE
we just trained in HARP. Math in plain text throughout.

**Goal.** Find a demand matrix d that the optimal routes with zero congestion
but DOTE congests badly. Baseline to beat: **1.9065**, DOTE's worst on 4032 real
test snapshots. Paper reports 6x (DOTE-hist) / 3.47x (DOTE-curr).

---

## What already exists

| Piece | Where | Paper |
|---|---|---|
| Frozen DOTE, differentiable end to end | `frameworks/dote_system.py` | H(x), Sec 3.2 |
| Gradient reaches the input TM | verified in `000_verify_dote.py` | "piecewise sub-differentiable", Sec 3.2 |
| paths-to-edges matrix, capacities | `DM_Dataset_within_Cluster` | pte, Fig 2 pipeline |
| Exact LP optimum on demand | `frameworks/gurobi_mlu.py` | MLU_OPT, Eq (2) |

Nothing new is needed on the DNN side. The search is new code around it.

---

## Steps

### 1. M_adv as a differentiable function of d  — Eq (2) -> Eq (3)

    MLU_DOTE(d) = max over edges of  [ pte^T (split_DOTE(d) * d_rep) ] / cap

`split_DOTE(d)` is the frozen network. Since the constraint will pin the
denominator to 1, the objective is just the numerator: `M_adv(d) = MLU_DOTE(d)`.

*Why:* Eq (2) is a ratio and non-convex; Eq (3) removes the quotient.

### 2. The optimal side, with f as a free variable  — Eq (3), the "exists f"

    MLU(d, f) = max over edges of  [ pte^T (f * d_rep) ] / cap

f is NOT obtained from Gurobi inside the loop. It is a search variable, so the
LP never has to be differentiated through.

*Why:* differentiating an argmin is the "not clean derivative" case the paper
routes around (Sec 4).

### 3. Lagrangian  — Eq (4)

    L(d, f, lam) = M_adv(d) + lam * (MLU(d, f) - 1)

*Why:* gradient ascent needs an unconstrained objective.

### 4. Gradient descent-ascent  — Eq (5)

    repeat:
      T times:  d   <- d   + a_d * grad_d L
                f   <- f   + a_f * grad_f L
      once:     lam <- lam - a_lam * grad_lam L        # grad_lam L = MLU(d,f) - 1

Paper's defaults: a_d = a_f = a_lam = 0.01, T = 1.

*Why:* inner max over (d, f), outer min over lam. Negative lam is what turns
ascent on f into descent on MLU - see mental-model.md Sec 6.

### 5. Feasibility of the variables  — not specified by the paper

- `d >= 0`: clamp after each step (projected ascent).
- `f` on the simplex per pair: parametrize `f = softmax(z)` and ascend on z.
- `lam`: free scalar, init 0.

*Flag:* these are our choices. The paper states neither. Worth recording since
they affect what the search can reach.

### 6. Honest reporting  — back to Eq (2)

Do **not** trust the relaxation to land exactly on MLU_OPT(d) = 1. After the
search converges:

1. take the final d,
2. solve the real LP with Gurobi -> MLU_OPT(d),
3. report `ratio = MLU_DOTE(d) / MLU_OPT(d)`.

*Why:* this is Eq (2) evaluated exactly, so the number is immune to constraint
violation in the Lagrangian. If the search cheated, this is where it shows.

### 7. Baselines  — Tables 1 and 2

- Random search over demands (the paper's black-box straw man, 1.22x / 1.25x).
- DOTE's worst real test snapshot: 1.9065.
- Report wall-clock (paper: ~50 s, vs 6 h for MetaOpt).

### 8. Robustness  — Sec 5 and Table 3

- Multiple random restarts; the objective is non-convex, so this is a lower
  bound (paper runs each experiment 5 times).
- Step-size sensitivity sweep over a_d, mirroring Table 3.
- Figure 5 analogue: compare the adversarial demand distribution against the
  training data.

---

## Deliverables

    frameworks/gray_box_search.py     steps 1-5, the search itself
    002_run_adversarial_search.py     driver: restarts, Gurobi check, reporting
    002_*.png / 002_*.txt             outputs (per the XXX_ rule)

## Known soft spots, carried from the paper

- Local optimum only: the DNN's non-convexity survives Eq (3). Any number we get
  is a lower bound on DOTE's worst case.
- At a kink the gradient never vanishes, so a fixed step size oscillates rather
  than settles - expect jitter near the peak, and prefer the best iterate seen
  over the last one.
- The subdifferential chain rule is an inclusion, not an equality.

## Open question before coding

The paper's 6x is for **DOTE-hist** (predicts from the previous 12 matrices).
Ours is **DOTE-curr** (`--pred 0`, routes what it is fed), which the paper
reports at **3.47x**. 3.47x is the number to compare against, not 6x.
