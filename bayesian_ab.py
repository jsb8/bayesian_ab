# %%
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
import scipy.integrate
import scipy.stats
from tqdm import tqdm


@dataclass
class TrialData(object):
    p1: float
    p2: float
    samples1: np.ndarray
    samples2: np.ndarray
    efficacy_prob: float
    futility_prob: float


"""Difference of two betas PDF + CDF"""
try:
    import mpmath as mp

    mp.mp.dps = 32

    def beta_diff_pdf(x, a1, b1, a2, b2):
        """PDF of Beta(a1, b1) - Beta(a2, b2) at x"""
        assert -1 <= x <= 1
        x = mp.mpf(x)
        A = mp.beta(a1, b1) * mp.beta(a2, b2)
        if x > 0:
            F = mp.hyper2d(
                {"m+n": [b1], "m": [a1 + b1 + a2 + b2 - 2], "n": [1 - a1]},
                {"m+n": [b1 + a2]},
                1 - x,
                1 - x * x,
            )
            return (
                mp.beta(a2, b1) * x ** (b1 + b2 - 1) * (1 - x) ** (a2 + b1 - 1) * F / A
            )
        elif x < 0:
            F = mp.hyper2d(
                {"m+n": [b2], "m": [1 - a2], "n": [a1 + b1 + a2 + b2 - 2]},
                {"m+n": [a1 + b2]},
                1 - x * x,
                1 + x,
            )
            return (
                mp.beta(a1, b2)
                * (-x) ** (b1 + b2 - 1)
                * (1 + x) ** (a1 + b2 - 1)
                * F
                / A
            )
        else:
            return mp.beta(a1 + a2 - 1, b1 + b2 - 1) / A

    def beta_diff_cdf(x, a1, b1, a2, b2):
        """CDF of Beta(a1, b1) - Beta(a2, b2) at x"""
        _beta_diff_pdf = partial(beta_diff_pdf, a1=a1, b1=b1, a2=a2, b2=b2)
        return scipy.integrate.quad(_beta_diff_pdf, -1.0, x)[0]


except:
    pass


"""Monte Carlo CDF

Manually install cupy to use GPU function
"""
try:
    import cupy

    def beta_diff_cdf_mc(x, a1, b1, a2, b2, n=int(1e5)):
        """Monte Carlo CDF of Beta(a1, b1) - Beta(a2, b2) at x"""
        A = cupy.random.beta(a1, b1, size=n, dtype=np.float64)
        B = cupy.random.beta(a2, b2, size=n, dtype=np.float64)
        if x != 0.0:
            I = A <= B + x
        else:
            I = A <= B
        ret = I.sum() / n
        return ret.get()


except:

    def beta_diff_cdf_mc(x, a1, b1, a2, b2, n=int(1e5)):
        """Monte Carlo CDF of Beta(a1, b1) - Beta(a2, b2) at x"""
        return (
            scipy.stats.beta.rvs(a1, b1, size=n)
            < scipy.stats.beta.rvs(a2, b2, size=n) + x
        ).mean()


# %%
# Set up simulation parameters
N = 1000  # Total number of trials
T = 500  # Number of possible samples per trial
K = 1  # Minimum observations
efficacy_confidence = 0.9
futility_confidence = 0.9
true_prior_alpha = 3
true_prior_beta = 5
true_prior = scipy.stats.beta(true_prior_alpha, true_prior_beta)
guess_prior_alpha = 3
guess_prior_beta = 5
guess_prior = scipy.stats.beta(guess_prior_alpha, guess_prior_beta)
trial_data = []

# %%
# Run simulation over N steps, this is not optimized for speed but
# written to be as clear as possible
for _ in tqdm(range(N)):
    # Draw the true p for each treatment
    p1 = true_prior.rvs()
    p2 = true_prior.rvs()

    # Start with a minimum of K samples
    all_samples1 = scipy.stats.bernoulli.rvs(p1, size=T)
    all_samples2 = scipy.stats.bernoulli.rvs(p2, size=T)

    t = K
    while True:
        # Subset the samples at step t
        samples1 = all_samples1[0:t]
        samples2 = all_samples2[0:t]

        # Compute posterior - Beta(alpha + success, beta + N - failure)
        posterior_alpha1 = guess_prior_alpha + samples1.sum()
        posterior_beta1 = guess_prior_beta + len(samples1) - samples1.sum()
        posterior_alpha2 = guess_prior_alpha + samples2.sum()
        posterior_beta2 = guess_prior_beta + len(samples2) - samples2.sum()

        # Compute P(p_{A} < p_{B} | D), where A is parameter in treatment A
        efficacy_prob = beta_diff_cdf_mc(
            0.0,
            posterior_alpha1,
            posterior_beta1,
            posterior_alpha2,
            posterior_beta2,
            n=int(1e4),
        )
        futility_prob = 1 - efficacy_prob
        # If we are close, compute with more accuracy
        if (efficacy_prob > efficacy_confidence - 0.03) or (
            futility_prob > futility_confidence - 0.03
        ):
            efficacy_prob = beta_diff_cdf_mc(
                0.0,
                posterior_alpha1,
                posterior_beta1,
                posterior_alpha2,
                posterior_beta2,
                n=int(1e7),
            )
            futility_prob = 1 - efficacy_prob

        # Check for trial termination
        if (
            (efficacy_prob > efficacy_confidence)
            or (futility_prob > futility_confidence)
            or (len(samples1) >= T)
        ):
            trial_data.append(
                TrialData(p1, p2, samples1, samples2, efficacy_prob, futility_prob)
            )
            break
        else:
            # Go to next sample
            t += 1


# %%
# Sort our trials
efficacy_trials = [
    t
    for t in trial_data
    if t.efficacy_prob > efficacy_confidence and len(t.samples1) < T
]
futile_trials = [
    t
    for t in trial_data
    if t.futility_prob > futility_confidence and len(t.samples1) < T
]
completed_trials = [t for t in trial_data if len(t.samples1) >= T]

# %%
# Print some statistics
print(f"N efficacy {len(efficacy_trials)}")
print(f"N futile {len(futile_trials)}")
print(f"N completed {len(completed_trials)}")
print(
    f"Total = {len(trial_data)} vs {len(efficacy_trials) + len(futile_trials) + len(completed_trials)}"
)

# %%
# Print some statistics
print(
    f"Efficacy P(p1 < p2) {np.mean([t.p1 < t.p2 for t in efficacy_trials])}"
    f" vs mean(efficacy) {np.mean([t.efficacy_prob for t in efficacy_trials])}"
)
print(
    f"Futile P(p1 > p2) {np.mean([t.p1 > t.p2 for t in futile_trials])}"
    f" vs mean(futile) {np.mean([t.futility_prob for t in futile_trials])}"
)

# %%
# Posterior means - This is mostly data wrangling to generate a table
all_trial_means = []
for trial_name, trial_type in [
    ("efficacy", efficacy_trials),
    ("futility", futile_trials),
    ("completed", completed_trials),
]:
    trial_means = []
    for t in tqdm(trial_type):
        posterior_alpha1 = guess_prior_alpha + t.samples1.sum()
        posterior_beta1 = guess_prior_beta + len(t.samples1) - t.samples1.sum()
        posterior_alpha2 = guess_prior_alpha + t.samples2.sum()
        posterior_beta2 = guess_prior_beta + len(t.samples2) - t.samples2.sum()
        trial_mean = pd.DataFrame(
            [
                [
                    t.p1,
                    t.samples1.mean(),
                    scipy.stats.beta(posterior_alpha1, posterior_beta1).mean(),
                ],
                [
                    t.p2,
                    t.samples2.mean(),
                    scipy.stats.beta(posterior_alpha2, posterior_beta2).mean(),
                ],
            ],
            columns=["true_prob", "observed_mean", "posterior_mean"],
            index=pd.Index(["A", "B"], name="treatment"),
        )
        trial_means.append(trial_mean)
    trial_means = pd.concat(trial_means)
    trial_means["group"] = trial_name
    trial_means = trial_means.set_index("group", append=True)
    all_trial_means.append(trial_means)

trial_means = pd.concat(all_trial_means)

# %%
# Display the table grouped by group and treatment
trial_means.groupby(["group", "treatment"]).mean()
