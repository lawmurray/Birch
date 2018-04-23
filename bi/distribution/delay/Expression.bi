/**
 * Delayed expression.
 *
 * - Value: Value type.
 */
class Expression<Value> < Delay {  
  /**
   * Memoized result of expression.
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
   * Node-specific value.
   */
  function doValue() -> Value {
    assert false;
  }
}
