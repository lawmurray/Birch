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
  ρ:Beta <- ρ;

  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_beta_binomial(n.value(), ρ.α.value(), ρ.β.value());
    }
  }
  
  function logpdf(x:Integer) -> Real {
    return logpdf_beta_binomial(x, n.value(), ρ.α.value(), ρ.β.value());
  }

  function update(x:Integer) {
    (ρ.α, ρ.β) <- box(update_beta_binomial(x, n.value(), ρ.α.value(), ρ.β.value()));
  }

  function downdate(x:Integer) {
    (ρ.α, ρ.β) <- box(downdate_beta_binomial(x, n.value(), ρ.α.value(), ρ.β.value()));
  }

  function cdf(x:Integer) -> Real? {
    return cdf_beta_binomial(x, n.value(), ρ.α.value(), ρ.β.value());
  }
  
  function lower() -> Integer? {
    return 0;
  }
  
  function upper() -> Integer? {
    return n.value();
  }

  function link() {
    ρ.setChild(this);
  }
  
  function unlink() {
    ρ.releaseChild(this);
  }
}

function BetaBinomial(n:Expression<Integer>, ρ:Beta) -> BetaBinomial {
  m:BetaBinomial(n, ρ);
  m.link();
  return m;
}
