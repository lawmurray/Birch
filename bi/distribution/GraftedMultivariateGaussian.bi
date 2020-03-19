/*
 * Grafted multivariate Gaussian distribution.
 */
class GraftedMultivariateGaussian(μ:Expression<Real[_]>,
    Σ:Expression<Real[_,_]>) < MultivariateGaussian(μ, Σ) {
  function graft() -> Distribution<Real[_]> {
    prune();
    graftFinalize();
    return this;
  }

  function graftMultivariateGaussian() -> MultivariateGaussian? {
    prune();
    graftFinalize();
    return this;
  }

  function graftFinalize() -> Boolean {
    μ.value();
    Σ.value();
    return true;
  }
}

function GraftedMultivariateGaussian(μ:Expression<Real[_]>,
    Σ:Expression<Real[_,_]>) -> GraftedMultivariateGaussian {
  m:GraftedMultivariateGaussian(μ, Σ);
  return m;
}
