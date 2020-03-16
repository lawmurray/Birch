/*
 * Beta-Bernoulli distribution.
 */
final class BetaBernoulli(ρ:Beta) < Distribution<Boolean> {
  /**
   * Success probability.
   */
  ρ:Beta& <- ρ;

  function simulate() -> Boolean {
    return simulate_beta_bernoulli(ρ.α.value(), ρ.β.value());
  }
  
  function logpdf(x:Boolean) -> Real {
    return logpdf_beta_bernoulli(x, ρ.α.value(), ρ.β.value());
  }

  function update(x:Boolean) {
    (ρ.α, ρ.β) <- update_beta_bernoulli(x, ρ.α.value(), ρ.β.value());
  }

  function downdate(x:Boolean) {
    (ρ.α, ρ.β) <- downdate_beta_bernoulli(x, ρ.α.value(), ρ.β.value());
  }

  function graftFinalize() -> Boolean {
    if !ρ.hasValue() {
      ρ.setChild(this);
      return true;
    } else {
      return false;
    }
  }
}

function BetaBernoulli(ρ:Beta) -> BetaBernoulli {
  m:BetaBernoulli(ρ);
  return m;
}
