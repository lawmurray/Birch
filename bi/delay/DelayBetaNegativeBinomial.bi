/*
 * Delayed Beta-negative-binomial random variate
 */
final class DelayBetaNegativeBinomial(future:Integer?, futureUpdate:Boolean,
    k:Integer, ρ:DelayBeta) < DelayDiscrete(future, futureUpdate) {
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

  function logpdf(x:Integer) -> Real {
    return logpdf_beta_negative_binomial(x, k, ρ!.α, ρ!.β);
  }

  function update(x:Integer) {
    (ρ!.α, ρ!.β) <- update_beta_negative_binomial(x, k, ρ!.α, ρ!.β);
  }

  function downdate(x:Integer) {
    (ρ!.α, ρ!.β) <- downdate_beta_negative_binomial(x, k, ρ!.α, ρ!.β);
  }

  function pdf(x:Integer) -> Real {
    return pdf_beta_negative_binomial(x, k, ρ!.α, ρ!.β);
  }
}

function DelayBetaNegativeBinomial(future:Integer?, futureUpdate:Boolean, k:Integer, ρ:DelayBeta) -> DelayBetaNegativeBinomial {
  m:DelayBetaNegativeBinomial(future, futureUpdate, k, ρ);
  ρ.setChild(m);
  return m;
}
