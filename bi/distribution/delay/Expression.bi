/**
 * Delayed expression.
 *
 * - Value: Value type.
 */
class Expression<Value> < Delay {  
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
  function getGaussian() -> (Real, Real) {
    assert false;
  }

  /**
   * If `isGaussian()`, set its parameters, otherwise undefined.
   */
  function setGaussian(θ:(Real, Real)) {
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
  function getAffineGaussian() -> (Real, Real, Real, Real) {
    assert false;
  }

  /**
   * If `isAffineGaussian()`, set its parameters, otherwise undefined.
   */
  function setAffineGaussian(θ:(Real, Real)) {
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
  function getBeta() -> (Real, Real) {
    assert false;
  }

  /**
   * If `isBeta()`, set its parameters, otherwise undefined.
   */
  function setBeta(θ:(Real, Real)) {
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
  function getGamma() -> (Real, Real) {
    assert false;
  }

  /**
   * If `isGamma()`, set its parameters, otherwise undefined.
   */
  function setGamma(θ:(Real, Real)) {
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
  function getInverseGamma() -> (Real, Real) {
    assert false;
  }

  /**
   * If `isInverseGamma()`, set its parameters, otherwise undefined.
   */
  function setInverseGamma(θ:(Real, Real)) {
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
  function getScaledInverseGamma(σ2:Expression<Real>) -> (Real, Real, Real) {
    assert false;
  }

  /**
   * If `isScaledInverseGamma()`, set its parameters, otherwise undefined.
   */
  function setScaledInverseGamma(σ2:Expression<Real>, θ:(Real, Real)) {
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
  function getNormalInverseGamma(σ2:Expression<Real>) -> (Real, Real, Real, Real) {
    assert false;
  }

  /**
   * If `isNormalInverseGamma()`, set its parameters, otherwise undefined.
   */
  function setNormalInverseGamma(σ2:Expression<Real>, θ:(Real, Real, Real, Real)) {
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
  function getAffineNormalInverseGamma(σ2:Expression<Real>) -> (Real, Real, Real, Real, Real, Real) {
    assert false;
  }

  /**
   * If `isAffineNormalInverseGamma()`, set its parameters, otherwise undefined.
   */
  function setAffineNormalInverseGamma(σ2:Expression<Real>, θ:(Real, Real, Real, Real)) {
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
  function getDirichlet() -> Real[_] {
    assert false;
  }

  /**
   * If `isDirichlet()`, set its parameters, otherwise undefined.
   */
  function setDirichlet(θ:Real[_]) {
    assert false;
  }

  /**
   * Node-specific value.
   */
  function doValue() -> Value {
    assert false;
  }
}
