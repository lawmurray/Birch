/*
 * Delayed Beta-negative-binomial random variate
 */
class DelayBetaNegativeBinomial(x:Random<Integer>&, k:Integer, ρ:DelayBeta) < DelayDiscrete(x) {
  /**
   * Number of successes before the experiment is stopped.
   */
  k:Integer <- k;

  /**
   * Success probability.
   */
  ρ:DelayBeta& <- ρ;

  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_beta_negative_binomial(k, ρ!.α, ρ!.β);
    }
  }

  function observe(x:Integer) -> Real {
    return observe_beta_negative_binomial(x, k, ρ!.α, ρ!.β);
  }

  function update(x:Integer) {
    (ρ!.α, ρ!.β) <- update_beta_negative_binomial(x, k, ρ!.α, ρ!.β);
  }

  function downdate(x:Integer) {
    (ρ!.α, ρ!.β) <- downdate_beta_negative_binomial(x, k, ρ!.α, ρ!.β);
  }

  function pmf(x:Integer) -> Real {
    return pmf_beta_negative_binomial(x, k, ρ!.α, ρ!.β);
  }
}

function DelayBetaNegativeBinomial(x:Random<Integer>&, k:Integer, ρ:DelayBeta) -> DelayBetaNegativeBinomial {
  m:DelayBetaNegativeBinomial(x, k, ρ);
  ρ.setChild(m);
  return m;
}
