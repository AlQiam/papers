# The storyline, and the math under it

*Namyar et al., "End-to-End Performance Analysis of Learning-enabled Systems,"
HotNets '24.*

**The whole paper is one chain of forced moves.** Each section exists because the
previous one left exactly one option open. Hold the chain and the paper is yours.

> **Question.** How badly can a deployed learning-enabled system underperform —
> not on its test set, but on the worst input that exists?

## 1. Why the DNN alone is the wrong object

DOTE's DNN emits split ratios; the operator cares about MLU. These are not the
same thing: Figure 3 shows two different split-ratio vectors producing *identical*
MLU, and the same vector producing different MLU under different demands. So a
bound on the DNN's output error implies nothing about system performance.

**Forced:** analyze $\mathcal{M}_{adv}(H(x))$ — an objective over the whole
pipeline $H$ = DNN + post-processor + routing + MLU.

## 2. Why the existing two families both fail

**White-box** (MetaOpt, Virelay) encodes the system as one MILP/SMT instance.
Two blockers: it needs piecewise-*linear* activations (the authors had to swap
DOTE's activation out just to run the comparison), and jointly encoding
components makes one component's output another's variable — non-convex,
non-linear, unsolvable at scale. It found nothing in 6 hours.

**Black-box** random search uses no structure, so it wanders. It found 1.22x.

**Forced:** use *partial* information — enough to get a direction, not enough to
need a global model. That is the gradient.

## 3. Why a gradient is available at all

This is the paper's enabling fact, and its most compressed sentence: *DNNs are
piecewise sub-differentiable.*

**Piecewise.** Fixing every ReLU's on/off state fixes a region of input space, and
on that region the network is affine, $H(x) = W_{\text{eff}}x + b_{\text{eff}}$.
The regions are convex polyhedra; the gradient is exact and constant inside each.

**Sub-differentiable.** On a region boundary there is no derivative, but there is
a non-empty set of slopes of lines touching the graph and staying below it:

$$\partial\sigma(0) = \{\,g : \sigma(z) \ge \sigma(0) + gz \ \ \forall z\,\} = [0,1]$$

Any element is a subgradient; autodiff picks one. (Strictly this is the *Clarke*
subdifferential — the convex-analysis definition above needs convex $f$, and DNNs
are not convex. Same $[0,1]$, weaker assumptions.)

**Why that is enough.** Write the system as a composition and apply the chain rule:

$$\nabla_x \mathcal{M}_{adv} = \nabla_y \mathcal{M}_{adv}\cdot\nabla_z H_2 \cdot \nabla_x H_1$$

Each $H_i$ is differentiated **in place** — analytically, or estimated from
samples of its input/output behavior. No component is ever written as a
closed-form expression for a solver. *That* is the gray-box saving: modeling one
component at a time is easy; modeling their joint dependencies is what killed
white-box.

## 4. Why the objective is a ratio, and why that hurts

Absolute MLU is not a measure of failure — scale every demand up and any system
congests. The meaningful quantity is distance from optimal:

$$\mathcal{M}_{adv}(d) \;\triangleq\; \frac{\mathrm{MLU}_{DOTE}(d)}{\mathrm{MLU}_{OPT}(d)} \;=\; \max_f \frac{\mathrm{MLU}_{DOTE}(d)}{\mathrm{MLU}(d,f)} \tag{2}$$

**Why this is non-convex in $d$**, twice over: the numerator is a non-convex DNN
composed with a max (ReLU is convex, but the next layer's weights may be
negative, and convexity survives composition only under a non-decreasing outer
function); and a quotient of convex functions is not convex anyway. Gradient
ascent on it reaches a local optimum only.

## 5. Why the denominator can simply be deleted

For a fixed routing $f$, every link load is linear in $d$, so

$$\mathrm{MLU}(\kappa d, f) = \kappa\,\mathrm{MLU}(d,f) \quad\Longrightarrow\quad \mathrm{MLU}_{OPT}(\kappa d) = \kappa\,\mathrm{MLU}_{OPT}(d)$$

Positive homogeneity. So **every ray of demands crosses the surface
$\mathrm{MLU}_{OPT}=1$ exactly once**, at $\kappa = 1/\mathrm{MLU}_{OPT}(d)$. That
surface is a cross-section of the demand cone: one representative per *direction*,
magnitude discarded. Restrict to it and the denominator is identically 1:

$$\mathcal{M}_{adv}(d) \triangleq \mathrm{MLU}_{DOTE}(d), \qquad d \in \{d \mid \exists f: \mathrm{MLU}(d,f)=1\} \tag{3}$$

**The hidden assumption.** This is lossless only if DOTE's split ratios are
unchanged by rescaling — then $\mathrm{MLU}_{DOTE}$ is homogeneous too and the
ratio is constant along each ray. A DNN has no reason to be scale-invariant, and
the authors only *condition* on it ("if DOTE's split ratios remain the same").
The fallback is to re-run the whole search over $\{d \mid \exists f: \mathrm{OPT}(d,f)=P\}$
for several $P$ and keep the best — restoring the magnitude dimension. For MLU,
$P=1$ suffices.

## 6. Why a Lagrangian, and what $\lambda$ actually does

Eq. (3) is constrained; gradient ascent needs an unconstrained objective. Fold
the constraint in as a penalty:

$$\min_\lambda \max_{d,f}\ \mathcal{L}_{final} \triangleq \mathcal{M}_{adv}(d) + \lambda\big(\mathrm{MLU}(d,f)-1\big) \tag{4}$$

The sign of $\lambda$ is the whole trick. The multiplier step is
$\lambda \leftarrow \lambda - \alpha_\lambda(\mathrm{MLU}(d,f)-1)$, so congestion
above 1 drives $\lambda$ negative. Since $\nabla_f\mathcal{L} = \lambda\nabla_f \mathrm{MLU}(d,f)$,
a negative $\lambda$ turns **ascent on $f$ into descent on MLU** — which is exactly
what the optimal routing must do. The formal $\max_f$ produces a *minimizing*
routing. Solved by multi-step gradient descent-ascent: $T$ ascent steps on
$(d,f)$, one descent step on $\lambda$ (Eq. 5).

## 7. What the search actually returns

A demand matrix that the optimal routes with **zero congestion**, together with
the routing $f$ that proves it, on which DOTE congests a link **6x** over. The
gap is therefore attributable to DOTE's decisions, not to a hard input — and
Figure 5 shows those demands look nothing like its training distribution. 50
seconds, against 6 fruitless hours for MetaOpt.

## The mental model, in one paragraph

*You cannot model the system, so don't: differentiate it. Piecewise
sub-differentiability makes every component locally linear, the chain rule glues
those local linearizations into an end-to-end direction, and you walk the input
uphill. Everything else is bookkeeping to make "uphill" well-posed — a ratio so
the metric means something, homogeneity to kill the denominator, a multiplier to
kill the constraint. The price is a local optimum and a single example.*

**Where it is thin:** the chain rule for subdifferentials is only an inclusion,
$\partial(f\circ g)\subseteq \partial f\cdot\partial g$; "estimate the gradient
from samples" carries real weight in the pitch but is never exercised, since
DOTE's pipeline happened to be differentiable throughout; and the scale-invariance
assumption in §5 is asserted conditionally, never established.
