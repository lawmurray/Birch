/**
 * Delayed expression.
 *
 * - Value: Value type.
 */
class Expression<Value> {  
  /**
   * Memoized result of evaluation.
   */
  x:Value?;

  /**
   * Value conversion.
   */
  operator -> Value {
    return value();
  }
  
  /**
   * Value conversion.
   */
  function value() -> Value {
    if (!x?) {
      x <- doValue();
      assert x?;
    }
    return x!;
  }
  
  /**
   * Observe the value.
   *
   * - x: The observed value.
   *
   * Returns: the log likelihood.
   */
  function observe(x:Value) -> Real {
    assert false;
  }
  
  /**
   * Are the values of any random variables within this expression missing?
   */
  function isMissing() -> Boolean {
    return !x?;
  }
  
  /**
   * Does this evaluate to a Gaussian distribution?
   */
  function isGaussian() -> Boolean {
    return false;
  }
  
  /**
   * If `isGaussian()`, get its parameters, otherwise undefined.
   */
  function getGaussian() -> DelayGaussian {
    assert false;
  }

  /**
   * Does this evaluate to an affine transformation of a Gaussian
   * distribution?
   */
  function isAffineGaussian() -> Boolean {
    return false;
  }
  
  /**
   * If `isAffineGaussian()`, get its parameters, otherwise undefined.
   */
  function getAffineGaussian() -> (Real, DelayGaussian, Real) {
    assert false;
  }

  /**
   * Does this evaluate to a Beta distribution?
   */
  function isBeta() -> Boolean {
    return false;
  }
  
  /**
   * If `isBeta()`, get its parameters, otherwise undefined.
   */
  function getBeta() -> DelayBeta {
    assert false;
  }

  /**
   * Does this evaluate to a Gamma distribution?
   */
  function isGamma() -> Boolean {
    return false;
  }
  
  /**
   * If `isGamma()`, get its parameters, otherwise undefined.
   */
  function getGamma() -> DelayGamma {
    assert false;
  }

  /**
   * Does this evaluate to an inverse Gamma distribution?
   */
  function isInverseGamma() -> Boolean {
    return false;
  }
  
  /**
   * If `isInverseGamma()`, get its parameters, otherwise undefined.
   */
  function getInverseGamma() -> DelayInverseGamma {
    assert false;
  } 

  /**
   * Does this evaluate to a scaled inverse gamma distribution with the given
   * inverse gamma marginal?
   */
  function isScaledInverseGamma(σ2:Expression<Real>) -> Boolean {
    return false;
  }
  
  /**
   * If `isScaledInverseGamma()`, get its parameters, otherwise undefined.
   */
  function getScaledInverseGamma(σ2:Expression<Real>) -> (Real, DelayInverseGamma) {
    assert false;
  }

  /**
   * Does this evaluate to a normal inverse gamma distribution with the given
   * inverse gamma marginal?
   */
  function isNormalInverseGamma(σ2:Expression<Real>) -> Boolean {
    return false;
  }
  
  /**
   * If `isNormalInverseGamma()`, get its parameters, otherwise undefined.
   */
  function getNormalInverseGamma(σ2:Expression<Real>) -> DelayNormalInverseGamma {
    assert false;
  }

  /**
   * Does this evaluate to an affine transformation of a normal inverse gamma
   * distribution with the given inverse gamma marginal?
   */
  function isAffineNormalInverseGamma(σ2:Expression<Real>) -> Boolean {
    return false;
  }
  
  /**
   * If `isAffineNormalInverseGamma()`, get its parameters, otherwise undefined.
   */
  function getAffineNormalInverseGamma(σ2:Expression<Real>) -> (Real, DelayNormalInverseGamma, Real) {
    assert false;
  }

  /**
   * Does this evaluate to a Dirichlet distribution?
   */
  function isDirichlet() -> Boolean {
    return false;
  }
  
  /**
   * If `isDirichlet()`, get its parameters, otherwise undefined.
   */
  function getDirichlet() -> DelayDirichlet {
    assert false;
  }

  /**
   * Does this evaluate to a multivariate Gaussian distribution?
   */
  function isMultivariateGaussian() -> Boolean {
    return false;
  }
  
  /**
   * If `iMultivariatesGaussian()`, get its parameters, otherwise undefined.
   */
  function getMultivariateGaussian() -> DelayMultivariateGaussian {
    assert false;
  }

  /**
   * Does this evaluate to an affine transformation of a multivariate Gaussian
   * distribution?
   */
  function isMultivariateAffineGaussian() -> Boolean {
    return false;
  }
  
  /**
   * If `isMultivariateAffineGaussian()`, get its parameters, otherwise undefined.
   */
  function getMultivariateAffineGaussian() -> (Real[_,_], DelayMultivariateGaussian, Real[_]) {
    assert false;
  }

  /**
   * Does this evaluate to a multivariate scaled inverse gamma distribution
   * with the given inverse gamma marginal?
   */
  function isMultivariateScaledInverseGamma(σ2:Expression<Real>) -> Boolean {
    return false;
  }
  
  /**
   * If `isMultivariateScaledInverseGamma()`, get its parameters, otherwise
   * undefined.
   */
  function getMultivariateScaledInverseGamma(σ2:Expression<Real>) ->
      (Real[_,_], DelayInverseGamma) {
    assert false;
  }

  /**
   * Does this evaluate to a multivariate normal inverse gamma distribution
   * with the given inverse gamma marginal?
   */
  function isMultivariateNormalInverseGamma(σ2:Expression<Real>) -> Boolean {
    return false;
  }
  
  /**
   * If `isMultivariateNormalInverseGamma()`, get its parameters, otherwise
   * undefined.
   */
  function getMultivariateNormalInverseGamma(σ2:Expression<Real>) ->
      DelayMultivariateNormalInverseGamma {
    assert false;
  }

  /**
   * Does this evaluate to an affine transformation of a multivariate normal
   * inverse gamma distribution with the given inverse gamma marginal?
   */
  function isMultivariateAffineNormalInverseGamma(σ2:Expression<Real>) -> Boolean {
    return false;
  }
  
  /**
   * If `isMultivariateAffineNormalInverseGamma()`, get its parameters,
   * otherwise undefined.
   */
  function getMultivariateAffineNormalInverseGamma(σ2:Expression<Real>) ->
      (Real[_,_], DelayMultivariateNormalInverseGamma, Real[_]) {
    assert false;
  }

  /**
   * Node-specific value.
   */
  function doValue() -> Value {
    assert false;
  }
}
