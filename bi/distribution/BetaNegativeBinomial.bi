/*
 * ed Beta-negative-binomial random variate
 */
final class BetaNegativeBinomial(k:Integer, ρ:Beta) < Discrete {
  /**
   * Number of successes before the experiment is stopped.
   */
  k:Integer <- k;

  /**
   * Success probability.
   */
  ρ:Beta& <- ρ;

  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_beta_negative_binomial(k, ρ.α, ρ.β);
    }
  }

  function logpdf(x:Integer) -> Real {
    return logpdf_beta_negative_binomial(x, k, ρ.α, ρ.β);
  }

  function update(x:Integer) {
    (ρ.α, ρ.β) <- update_beta_negative_binomial(x, k, ρ.α, ρ.β);
  }

  function downdate(x:Integer) {
    (ρ.α, ρ.β) <- downdate_beta_negative_binomial(x, k, ρ.α, ρ.β);
  }
  
  function lower() -> Integer? {
    return 0;
  }
}

function BetaNegativeBinomial(k:Integer, ρ:Beta) -> BetaNegativeBinomial {
  m:BetaNegativeBinomial(k, ρ);
  ρ.setChild(m);
  return m;
}
