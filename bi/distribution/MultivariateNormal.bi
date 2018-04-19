/**
 * Synonym for MultivariateGaussian.
 */
class MultivariateNormal<Type1,Type2> = MultivariateGaussian<Type1,Type2>;

/**
 * Create multivariate Gaussian distribution.
 */
function Normal(μ:Real[_], Σ:Real[_,_]) ->
    MultivariateGaussian<Real[_],Real[_,_]> {
  return Gaussian(μ, Σ);
}

/**
 * Create multivariate Gaussian distribution.
 */
function Normal(μ:Expression<Real[_]>, Σ:Real[_,_]) ->
    MultivariateGaussian<Expression<Real[_]>,Real[_,_]> {
  return Gaussian(μ, Σ);
}

/**
 * Create multivariate Gaussian distribution.
 */
function Normal(μ:Real[_], Σ:Expression<Real[_,_]>) ->
    MultivariateGaussian<Real[_],Expression<Real[_,_]>> {
  return Gaussian(μ, Σ);
}

/**
 * Create multivariate Gaussian distribution.
 */
function Normal(μ:Expression<Real[_]>, Σ:Expression<Real[_,_]>) ->
    MultivariateGaussian<Expression<Real[_]>,Expression<Real[_,_]>> {
  return Gaussian(μ, Σ);
}
