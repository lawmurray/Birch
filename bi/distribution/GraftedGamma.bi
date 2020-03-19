/*
 * Grafted gamma distribution.
 */
final class GraftedGamma(k:Expression<Real>, θ:Expression<Real>) <
    Gamma(k, θ) {
  function graft() -> Distribution<Real> {
    if !hasValue() {
      prune();
      graftFinalize();
    }
    return this;
  }

  function graftGamma() -> Gamma? {
    if !hasValue() {
      prune();
      graftFinalize();
      return this;
    } else {
      return nil;
    }
  }

  function graftFinalize() -> Boolean {
    k.value();
    θ.value();
    return true;
  }
}

function GraftedGamma(k:Expression<Real>, θ:Expression<Real>) ->
    GraftedGamma {
  m:GraftedGamma(k, θ);
  return m;
}
