/*
 * Grafted multivariate Gaussian distribution.
 */
class GraftedMultivariateGaussian(μ:Expression<Real[_]>,
    Σ:Expression<Real[_,_]>) < MultivariateGaussian(μ, Σ) {
  function graft() -> Distribution<Real[_]> {
    prune();
    return this;
  }

  function graftMultivariateGaussian() -> MultivariateGaussian? {
    prune();
    return this;
  }
}

function GraftedMultivariateGaussian(μ:Expression<Real[_]>,
    Σ:Expression<Real[_,_]>) -> GraftedMultivariateGaussian {
  m:GraftedMultivariateGaussian(μ, Σ);
  return m;
}
