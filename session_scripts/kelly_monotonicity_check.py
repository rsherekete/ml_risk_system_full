"""Empirical self-check for the fractional-Kelly book-assignment redesign.

Synthetic population of accounts with random per-account edge (mu), own P&L
volatility (sigma), and notional. Reproduces the exact retention formula
proposed for `book_assignment.py`:

    full_kelly_retain_i = clip((mu_i - markup_rate) / sigma_i**2, 0, cap)   [Kelly-eligible only]
    retain_i(kappa)     = kappa * full_kelly_retain_i        (Kelly-eligible accounts)
                        = fixed_i                              (insufficient-data / override accounts, kappa-independent)

kappa in [0, 1] is the fractional-Kelly dial (0 = never retain any Kelly-eligible
risk i.e. hedge everything; 1 = full Kelly).

We sweep kappa and check, with real numbers:
  1. total expected firm P&L(kappa)      -- must be non-decreasing (proof: linear, slope = v . excess_edge >= 0)
  2. undiversified portfolio risk(kappa) -- must be non-decreasing (proof: per-account square, disjoint support -> C2*kappa^2 + C0)
  3. diversified (correlation-aware) portfolio risk(kappa) -- reported empirically; this is
     NOT part of the formal guarantee once a kappa-independent "fixed" retention sleeve
     (insufficient-data accounts) coexists with the Kelly sleeve, because their cross-covariance
     term is not sign-constrained. We measure how often/how badly it is violated across many
     random seeds, and separately confirm it is *always* monotonic when there is no fixed sleeve
     (pure single-scalar-times-fixed-vector case), isolating exactly where the risk lives.
"""
import numpy as np

rng = np.random.default_rng(42)


def run_once(seed, n_accounts=500, insufficient_frac=0.10, override_frac=0.03,
             kappas=np.linspace(0, 1, 11), cap=1.0, net_markup_bps=1.0, verbose=False):
    r = np.random.default_rng(seed)
    notional = r.uniform(1_000, 200_000, n_accounts)
    markup_rate = net_markup_bps / 10_000.0
    markup_usd = notional * markup_rate

    # client's own predicted next-day P&L, expressed as a RATE per $1 notional
    # (this is what the new regression model `fit_oos_expected_pnl` predicts,
    # converted to a rate; mean slightly negative -> typical retail base rate).
    client_mu_rate = r.normal(loc=-0.0005, scale=0.0020, size=n_accounts)
    client_sigma_rate = r.uniform(0.005, 0.05, size=n_accounts)

    firm_edge_raw_rate = -client_mu_rate           # firm's edge rate from FULL retention
    excess_edge_rate = firm_edge_raw_rate - markup_rate   # edge over the hedge/markup alternative

    insufficient = r.random(n_accounts) < insufficient_frac
    override = r.random(n_accounts) < override_frac
    eligible = (excess_edge_rate > 0) & ~insufficient & ~override

    f_star = np.zeros(n_accounts)
    f_star[eligible] = excess_edge_rate[eligible] / (client_sigma_rate[eligible] ** 2)
    full_kelly_retain = np.clip(f_star, 0.0, cap)  # v_i -- fixed, kappa-independent target
    full_kelly_retain[~eligible] = 0.0

    fixed_retain = np.zeros(n_accounts)
    fixed_retain[insufficient] = 1.0    # legacy B_BOOK-by-default behaviour, kappa-independent
    fixed_retain[override] = 0.0        # persistent-edge/arbitrage override -> always fully hedged

    account_pnl_sigma_usd = client_sigma_rate * notional  # $ stdev of account P&L at full retention
    firm_edge_raw_usd = firm_edge_raw_rate * notional      # $ edge at full retention

    # random correlation matrix with genuine positive AND negative entries,
    # built from a low-rank factor model (realistic: shared market-direction factor)
    # plus idiosyncratic noise, so it is guaranteed PSD.
    n_factors = 5
    loadings = r.normal(0, 1, size=(n_accounts, n_factors))
    factor_cov = np.eye(n_factors)
    cov_raw = loadings @ factor_cov @ loadings.T + np.diag(r.uniform(0.5, 1.5, n_accounts))
    d = np.sqrt(np.diag(cov_raw))
    corr = cov_raw / np.outer(d, d)

    rows = []
    for kappa in kappas:
        retain = kappa * full_kelly_retain + fixed_retain
        hedge = 1.0 - retain
        expected_pnl = float(np.sum(hedge * markup_usd + retain * firm_edge_raw_usd))

        contrib_sigma = retain * account_pnl_sigma_usd
        undiv_variance = float(np.sum(contrib_sigma ** 2))

        cov_matrix = np.outer(account_pnl_sigma_usd, account_pnl_sigma_usd) * corr
        div_variance = float(retain @ cov_matrix @ retain)

        rows.append(dict(kappa=kappa, expected_pnl=expected_pnl,
                          undiv_variance=undiv_variance, div_variance=div_variance,
                          n_eligible=int(eligible.sum()), n_insufficient=int(insufficient.sum()),
                          n_override=int(override.sum())))
    return rows


def is_nondecreasing(values, tol=1e-6):
    values = np.asarray(values, dtype=float)
    diffs = np.diff(values)
    return bool(np.all(diffs >= -tol * max(1.0, np.max(np.abs(values)))))


# ---------------------------------------------------------------------------
# 1. One concrete, printed run.
# ---------------------------------------------------------------------------
print("=" * 78)
print("REPRESENTATIVE RUN (seed=42, 500 accounts)")
print("=" * 78)
rows = run_once(seed=42)
print(f"{'kappa':>6} {'expected_pnl_usd':>18} {'undiv_risk_std_usd':>20} {'div_risk_std_usd':>18}")
for row in rows:
    print(f"{row['kappa']:6.2f} {row['expected_pnl']:18,.0f} "
          f"{row['undiv_variance']**0.5:20,.0f} {row['div_variance']**0.5:18,.0f}")

pnl_series = [row["expected_pnl"] for row in rows]
undiv_series = [row["undiv_variance"] for row in rows]
div_series = [row["div_variance"] for row in rows]

print()
print(f"expected P&L monotonic non-decreasing:            {is_nondecreasing(pnl_series)}")
print(f"undiversified risk monotonic non-decreasing:       {is_nondecreasing(undiv_series)}")
print(f"diversified (correlated) risk monotonic non-dec.:  {is_nondecreasing(div_series)}")
print(f"n_eligible={rows[0]['n_eligible']} n_insufficient={rows[0]['n_insufficient']} n_override={rows[0]['n_override']}")

# ---------------------------------------------------------------------------
# 2. Stress test across many random seeds/scales, WITH the fixed
#    (insufficient-data) sleeve present -- this is the case that is NOT
#    formally guaranteed for the diversified risk measure.
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("STRESS TEST: 500 random seeds, WITH fixed insufficient-data sleeve")
print("=" * 78)
n_trials = 500
pnl_ok = 0
undiv_ok = 0
div_ok = 0
worst_div_violation = 0.0
for seed in range(n_trials):
    trial_rows = run_once(seed=seed, n_accounts=200, insufficient_frac=0.10, override_frac=0.03)
    pnl_s = [x["expected_pnl"] for x in trial_rows]
    undiv_s = [x["undiv_variance"] for x in trial_rows]
    div_s = [x["div_variance"] for x in trial_rows]
    pnl_ok += is_nondecreasing(pnl_s)
    undiv_ok += is_nondecreasing(undiv_s)
    div_ok += is_nondecreasing(div_s)
    diffs = np.diff(div_s)
    if diffs.min() < 0:
        worst_div_violation = min(worst_div_violation, diffs.min() / max(1.0, max(div_s)))

print(f"expected P&L monotonic in {pnl_ok}/{n_trials} seeds  (must be {n_trials}/{n_trials})")
print(f"undiversified risk monotonic in {undiv_ok}/{n_trials} seeds  (must be {n_trials}/{n_trials})")
print(f"diversified risk monotonic in {div_ok}/{n_trials} seeds  (empirical only -- NOT formally guaranteed)")
print(f"worst single-step diversified-risk relative violation observed: {worst_div_violation:.4%}")

# ---------------------------------------------------------------------------
# 3. Same stress test but with NO fixed sleeve at all (insufficient_frac=0,
#    override_frac=0) -- pure "kappa * fixed vector" case. Proof predicts this
#    must be monotonic even for diversified risk, for every seed.
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("STRESS TEST: 500 random seeds, NO fixed sleeve (pure kappa * v case)")
print("=" * 78)
pnl_ok2 = 0
undiv_ok2 = 0
div_ok2 = 0
for seed in range(n_trials):
    trial_rows = run_once(seed=seed + 10_000, n_accounts=200, insufficient_frac=0.0, override_frac=0.0)
    pnl_s = [x["expected_pnl"] for x in trial_rows]
    undiv_s = [x["undiv_variance"] for x in trial_rows]
    div_s = [x["div_variance"] for x in trial_rows]
    pnl_ok2 += is_nondecreasing(pnl_s)
    undiv_ok2 += is_nondecreasing(undiv_s)
    div_ok2 += is_nondecreasing(div_s)

print(f"expected P&L monotonic in {pnl_ok2}/{n_trials} seeds")
print(f"undiversified risk monotonic in {undiv_ok2}/{n_trials} seeds")
print(f"diversified risk monotonic in {div_ok2}/{n_trials} seeds  (proof predicts {n_trials}/{n_trials})")

# ---------------------------------------------------------------------------
# 4. Reproduce the ORIGINAL BUG MECHANISM for contrast: bare-probability
#    threshold shifting membership discontinuously, decoupled from $ edge
#    magnitude, CAN make total expected P&L non-monotonic in the "more
#    profit-seeking" direction -- demonstrating why the old design failed
#    where the new one cannot.
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("CONTRAST: reproducing the OLD threshold-flip bug mechanism")
print("=" * 78)
r = np.random.default_rng(7)
n = 300
notional = r.uniform(1_000, 200_000, n)
model_probability_loss = r.uniform(0, 1, n)
# a marginal account's true edge is NOT monotonically related to its predicted
# loss-probability rank in this synthetic illustration (deliberately, to mirror
# how a classifier's probability isn't the same object as a dollar edge) --
# accounts near probability ~0.3-0.5 include some of the single BIGGEST-notional,
# most profitable-to-retain accounts, so tightening the threshold below 0.5 first
# excludes exactly those.
true_edge_usd = (0.5 - model_probability_loss) * notional * 0.001 + r.normal(0, 50, n)

for profit_weight in [0, 20, 40, 60, 80, 100]:
    alpha = profit_weight / 100.0
    loss_threshold = float(np.clip(0.5 + 0.3 * (1 - 2 * alpha), 0.2, 0.8))
    a_book = model_probability_loss < loss_threshold
    # B_BOOK default: firm retains the rest. Old firm P&L = markup(A) - client_pnl(B).
    # Using true_edge_usd as the firm's edge-if-retained proxy here:
    b_book_pnl = float(np.sum(true_edge_usd[~a_book]))
    print(f"profit_weight={profit_weight:3d}  loss_threshold={loss_threshold:.2f}  "
          f"n_a_book={int(a_book.sum()):3d}  b_book_retained_edge_usd={b_book_pnl:12,.0f}")
