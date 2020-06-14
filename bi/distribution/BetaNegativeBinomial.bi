/**
 * Beta-negative-binomial distribution.
 */
final class BetaNegativeBinomial(k:Expression<Integer>, ρ:Beta) < Discrete {
  /**
   * Number of successes before the experiment is stopped.
   */
  k:Expression<Integer> <- k;

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
      return simulate_beta_negative_binomial(k.value(), ρ.α.value(), ρ.β.value());
    }
  }

  function simulateLazy() -> Integer? {
    if value? {
      return value!;
    } else {
      return simulate_beta_negative_binomial(k.get(), ρ.α.get(), ρ.β.get());
    }
  }

  function logpdf(x:Integer) -> Real {
    return logpdf_beta_negative_binomial(x, k.value(), ρ.α.value(), ρ.β.value());
  }

  function logpdfLazy(x:Expression<Integer>) -> Expression<Real>? {
    return logpdf_lazy_beta_negative_binomial(x, k, ρ.α, ρ.β);
  }

  function update(x:Integer) {
    (ρ.α, ρ.β) <- box(update_beta_negative_binomial(x, k.value(), ρ.α.value(), ρ.β.value()));
  }

  function updateLazy(x:Expression<Integer>) {
    (ρ.α, ρ.β) <- update_lazy_beta_negative_binomial(x, k, ρ.α, ρ.β);
  }

  function downdate(x:Integer) {
    (ρ.α, ρ.β) <- box(downdate_beta_negative_binomial(x, k.value(), ρ.α.value(), ρ.β.value()));
  }
  
  function lower() -> Integer? {
    return 0;
  }

  function link() {
    ρ.setChild(this);
  }
  
  function unlink() {
    ρ.releaseChild(this);
  }
}

function BetaNegativeBinomial(k:Expression<Integer>, ρ:Beta) ->
    BetaNegativeBinomial {
  m:BetaNegativeBinomial(k, ρ);
  m.link();
  return m;
}
