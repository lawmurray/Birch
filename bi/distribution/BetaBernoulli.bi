/**
 * Beta-Bernoulli distribution.
 */
final class BetaBernoulli(ρ:Beta) < Distribution<Boolean> {
  /**
   * Success probability.
   */
  ρ:Beta& <- ρ;

  function supportsLazy() -> Boolean {
    return true;
  }

  function simulate() -> Boolean {
    return simulate_beta_bernoulli(ρ.α.value(), ρ.β.value());
  }
  
  function simulateLazy() -> Boolean? {
    return simulate_beta_bernoulli(ρ.α.get(), ρ.β.get());
  }
  
  function logpdf(x:Boolean) -> Real {
    return logpdf_beta_bernoulli(x, ρ.α.value(), ρ.β.value());
  }

  function logpdfLazy(x:Expression<Boolean>) -> Expression<Real>? {
    return logpdf_lazy_beta_bernoulli(x, ρ.α, ρ.β);
  }

  function update(x:Boolean) {
    (ρ.α, ρ.β) <- box(update_beta_bernoulli(x, ρ.α.value(), ρ.β.value()));
  }

  function updateLazy(x:Expression<Boolean>) {
    (ρ.α, ρ.β) <- update_lazy_beta_bernoulli(x, ρ.α, ρ.β);
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
