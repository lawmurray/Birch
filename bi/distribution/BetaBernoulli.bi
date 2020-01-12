/*
 * Beta-Bernoulli distribution.
 */
final class BetaBernoulli(ρ:Beta) < Distribution<Boolean> {
  /**
   * Success probability.
   */
  ρ:Beta& <- ρ;

  function simulate() -> Boolean {
    return simulate_beta_bernoulli(ρ.α, ρ.β);
  }
  
  function logpdf(x:Boolean) -> Real {
    return logpdf_beta_bernoulli(x, ρ.α, ρ.β);
  }

  function update(x:Boolean) {
    (ρ.α, ρ.β) <- update_beta_bernoulli(x, ρ.α, ρ.β);
  }

  function downdate(x:Boolean) {
    (ρ.α, ρ.β) <- downdate_beta_bernoulli(x, ρ.α, ρ.β);
  }
}

function BetaBernoulli(ρ:Beta) -> BetaBernoulli {
  m:BetaBernoulli(ρ);
  ρ.setChild(m);
  return m;
}
