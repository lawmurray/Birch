/**
 * Beta-binomial distribution.
 */
final class BetaBinomial(n:Expression<Integer>, ρ:Beta) < BoundedDiscrete {
  /**
   * Number of trials.
   */
  n:Expression<Integer> <- n;

  /**
   * Success probability.
   */
  ρ:Beta& <- ρ;

  function supportsLazy() -> Boolean {
    return true;
  }

  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      auto ρ <- this.ρ;
      return simulate_beta_binomial(n.value(), ρ.α.value(), ρ.β.value());
    }
  }
  
  function simulateLazy() -> Integer? {
    if value? {
      return value!;
    } else {
      auto ρ <- this.ρ;
      return simulate_beta_binomial(n.get(), ρ.α.get(), ρ.β.get());
    }
  }
  
  function logpdf(x:Integer) -> Real {
    auto ρ <- this.ρ;
    return logpdf_beta_binomial(x, n.value(), ρ.α.value(), ρ.β.value());
  }

  function logpdfLazy(x:Expression<Integer>) -> Expression<Real>? {
    auto ρ <- this.ρ;
    return logpdf_lazy_beta_binomial(x, n, ρ.α, ρ.β);
  }

  function update(x:Integer) {
    auto ρ <- this.ρ;
    (ρ.α, ρ.β) <- box(update_beta_binomial(x, n.value(), ρ.α.value(), ρ.β.value()));
  }

  function updateLazy(x:Expression<Integer>) {
    auto ρ <- this.ρ;
    (ρ.α, ρ.β) <- update_lazy_beta_binomial(x, n, ρ.α, ρ.β);
  }

  function downdate(x:Integer) {
    auto ρ <- this.ρ;
    (ρ.α, ρ.β) <- box(downdate_beta_binomial(x, n.value(), ρ.α.value(), ρ.β.value()));
  }

  function cdf(x:Integer) -> Real? {
    auto ρ <- this.ρ;
    return cdf_beta_binomial(x, n.value(), ρ.α.value(), ρ.β.value());
  }
  
  function lower() -> Integer? {
    return 0;
  }
  
  function upper() -> Integer? {
    return n.value();
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

function BetaBinomial(n:Expression<Integer>, ρ:Beta) -> BetaBinomial {
  m:BetaBinomial(n, ρ);
  m.link();
  return m;
}
