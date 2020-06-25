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
    auto ρ <- this.ρ;
    return simulate_beta_bernoulli(ρ.α.value(), ρ.β.value());
  }
  
  function simulateLazy() -> Boolean? {
    auto ρ <- this.ρ;
    return simulate_beta_bernoulli(ρ.α.get(), ρ.β.get());
  }
  
  function logpdf(x:Boolean) -> Real {
    auto ρ <- this.ρ;
    return logpdf_beta_bernoulli(x, ρ.α.value(), ρ.β.value());
  }

  function logpdfLazy(x:Expression<Boolean>) -> Expression<Real>? {
    auto ρ <- this.ρ;
    return logpdf_lazy_beta_bernoulli(x, ρ.α, ρ.β);
  }

  function update(x:Boolean) {
    auto ρ <- this.ρ;
    (ρ.α, ρ.β) <- box(update_beta_bernoulli(x, ρ.α.value(), ρ.β.value()));
  }

  function updateLazy(x:Expression<Boolean>) {
    auto ρ <- this.ρ;
    (ρ.α, ρ.β) <- update_lazy_beta_bernoulli(x, ρ.α, ρ.β);
  }

  function downdate(x:Boolean) {
    auto ρ <- this.ρ;
    (ρ.α, ρ.β) <- box(downdate_beta_bernoulli(x, ρ.α.value(), ρ.β.value()));
  }

  function link() {
    auto ρ <- this.ρ;
    ρ.setChild(this);
  }
  
  function unlink() {
    auto ρ <- this.ρ;
    ρ.releaseChild(this);
  }
}

function BetaBernoulli(ρ:Beta) -> BetaBernoulli {
  m:BetaBernoulli(ρ);
  m.link();
  return m;
}
