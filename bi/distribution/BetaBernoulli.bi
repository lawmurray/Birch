/**
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
    (ρ.α, ρ.β) <- box(update_beta_bernoulli(x, ρ.α.value(), ρ.β.value()));
  }

  function downdate(x:Boolean) {
    (ρ.α, ρ.β) <- box(downdate_beta_bernoulli(x, ρ.α.value(), ρ.β.value()));
  }

  function link() {
    ρ.setChild(this);
  }
  
  function unlink() {
    ρ.releaseChild(this);
  }
}

function BetaBernoulli(ρ:Beta) -> BetaBernoulli {
  m:BetaBernoulli(ρ);
  m.link();
  return m;
}
