/*
 * Grafted Gaussian distribution.
 */
class GraftedGaussian(μ:Expression<Real>, σ2:Expression<Real>) <
    Gaussian(μ, σ2) {
  function graft() -> Distribution<Real> {
    prune();
    return this;
  }

  function graftGaussian() -> Gaussian? {
    prune();
    return this;
  }

  function graftNormalInverseGamma() -> NormalInverseGamma? {
    return nil;
  }
}

function GraftedGaussian(μ:Expression<Real>, σ2:Expression<Real>) ->
    GraftedGaussian {
  o:GraftedGaussian(μ, σ2);
  return o;
}
