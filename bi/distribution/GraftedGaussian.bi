/*
 * Grafted Gaussian distribution.
 */
class GraftedGaussian(μ:Expression<Real>, σ2:Expression<Real>) <
    Gaussian(μ, σ2) {
  function graft() -> Distribution<Real> {
    if !hasValue() {
      prune();
      graftFinalize();
    }
    return this;
  }

  function graftGaussian() -> Gaussian? {
    if !hasValue() {
      prune();
      graftFinalize();
      return this;
    } else {
      return nil;
    }
  }

  function graftNormalInverseGamma(compare:Distribution<Real>) ->
      NormalInverseGamma? {
    return nil;
  }

  function graftFinalize() -> Boolean {
    μ.value();
    σ2.value();
    return true;
  }
}

function GraftedGaussian(μ:Expression<Real>, σ2:Expression<Real>) ->
    GraftedGaussian {
  o:GraftedGaussian(μ, σ2);
  return o;
}
