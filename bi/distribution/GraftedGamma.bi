/*
 * Grafted gamma distribution.
 */
final class GraftedGamma(k:Expression<Real>, θ:Expression<Real>) <
    Gamma(k, θ) {
  function graft() -> Distribution<Real> {
    prune();
    return this;
  }

  function graftGamma() -> Gamma? {
    prune();
    return this;
  }
}

function GraftedGamma(k:Expression<Real>, θ:Expression<Real>) ->
    GraftedGamma {
  m:GraftedGamma(k, θ);
  return m;
}
