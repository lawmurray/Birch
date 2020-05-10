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
  ρ:Beta <- ρ;

  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_beta_negative_binomial(k.value(), ρ.α.value(), ρ.β.value());
    }
  }

  function logpdf(x:Integer) -> Real {
    return logpdf_beta_negative_binomial(x, k.value(), ρ.α.value(), ρ.β.value());
  }

  function update(x:Integer) {
    (ρ.α, ρ.β) <- update_beta_negative_binomial(x, k.value(), ρ.α.value(), ρ.β.value());
  }

  function downdate(x:Integer) {
    (ρ.α, ρ.β) <- downdate_beta_negative_binomial(x, k.value(), ρ.α.value(), ρ.β.value());
  }
  
  function lower() -> Integer? {
    return 0;
  }

  function link() {
    ρ.setChild(this);
  }
  
  function unlink() {
    ρ.releaseChild();
  }
}

function BetaNegativeBinomial(k:Expression<Integer>, ρ:Beta) ->
    BetaNegativeBinomial {
  m:BetaNegativeBinomial(k, ρ);
  m.link();
  return m;
}
