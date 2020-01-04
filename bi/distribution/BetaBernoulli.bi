/*
 * ed Beta-bernoulli random variate.
 */
final class BetaBernoulli(future:Boolean?, futureUpdate:Boolean, ρ:Beta) <
    Distribution<Boolean>(future, futureUpdate) {
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

function BetaBernoulli(future:Boolean?, futureUpdate:Boolean, ρ:Beta) ->
    BetaBernoulli {
  m:BetaBernoulli(future, futureUpdate, ρ);
  ρ.setChild(m);
  return m;
}
