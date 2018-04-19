/**
 * Synonym for Gaussian.
 */
class Normal<Type1,Type2> = Gaussian<Type1,Type2>;

/**
 * Create Gaussian distribution.
 */
function Normal(μ:Real, σ2:Real) -> Gaussian<Real,Real> {
  return Gaussian(μ, σ2);
}

/**
 * Create Gaussian distribution.
 */
function Normal(μ:Expression<Real>, σ2:Real) ->
    Gaussian<Expression<Real>,Real> {
  return Gaussian(μ, σ2);
}

/**
 * Create Gaussian distribution.
 */
function Normal(μ:Real, σ2:Expression<Real>) ->
    Gaussian<Real,Expression<Real>> {
  return Gaussian(μ, σ2);
}

/**
 * Create Gaussian distribution.
 */
function Normal(μ:Expression<Real>, σ2:Expression<Real>) ->
    Gaussian<Expression<Real>,Expression<Real>> {
  return Gaussian(μ, σ2);
}
