/*
 * Type-specific interface for delayed sampling $M$-path nodes.
 *
 * - Value: Value type.
 *
 * - future: Future value.
 * - futureUpdate: When realized, should the future value trigger an
 *   update? (Otherwise a downdate.)
 */
abstract class Distribution<Value>(future:Value?, futureUpdate:Boolean) < Delay {
  /**
   * Final value.
   */
  x:Value? <- nil;

  /**
   * Future value. This is set for situations where delayed sampling
   * is used, but when ultimately realized, a particular value (this one)
   * should be assigned, and updates or downdates applied accordingly. It
   * is typically used when replaying traces.
   */
  future:Value? <- future;

  /**
   * When assigned, should the future value trigger an update? (Otherwise
   * a downdate.)
   */
  futureUpdate:Boolean <- futureUpdate;

  /**
   * Number of rows, when interpreted as a matrix.
   */
  function rows() -> Integer {
    return 1;
  }

  /**
   * Number of columns, when interpreted as a matrix.
   */
  function columns() -> Integer {
    return 1;
  }

  /**
   * Does the node have a value?
   */
  function hasValue() -> Boolean {
    return x?;
  }

  /**
   * Realize a value for a random variate associated with this node.
   */
  function value() -> Value {
    if !x? {
      prune();
      if future? {
        x <- future!;
      } else {
        x <- simulate();
      }
    }
    return x!;
  }
  
  function pilot() -> Value {
    return value();
  }

  function propose() -> Value {
    return value();
  }

  function gradPilot(d:Value) -> Boolean {
    return false;
  }

  function gradPropose(d:Value) -> Boolean {
    return false;
  }

  function ratio() -> Real {
    return 0.0;
  }
  
  function accept() {
    //
  }

  function reject() {
    //
  }

  function clamp() {
    //
  }
  
  /**
   * Set value.
   */
  function set(x:Value) {
    assert !this.x?;
    assert !this.future?;
    this.x <- x;
    this.child <- nil;
  }

  /**
   * Assume the distribution for a random variate. When a value for the
   * random variate is required, it will be simulated from this distribution
   * and trigger an *update* on the delayed sampling graph.
   *
   * - v: The random variate.
   */
  function assume(v:Random<Value>) {
    assert !v.hasDistribution();
    assert !v.hasValue();
    
    futureUpdate <- true;
    v.p <- this;
  }

  /**
   * Assume this distribution for a random variate. When a value for the
   * random variate is required, it will be assigned according to the
   * `future` value given here, and trigger an *update* on the delayed
   * sampling graph.
   *
   * - v: The random variate.
   * - future: The future value.
   */
  function assume(v:Random<Value>, future:Value) {
    assert !v.hasDistribution();
    assert !v.hasValue();
    
    this.future <- future;
    futureUpdate <- true;
    v.p <- this;
  }

  /**
   * Assume the distribution for a random variate. When a value for the
   * random variate is required, it will be simulated from this distribution
   * and trigger an *downdate* on the delayed sampling graph.
   *
   * - v: The random variate.
   */
  function assumeWithDowndate(v:Random<Value>) {
    assert !v.hasDistribution();
    assert !v.hasValue();
    
    futureUpdate <- false;
    v.p <- this;
  }

  /**
   * Assume this distribution for a random variate. When a value for the
   * random variate is required, it will be assigned according to the
   * `future` value given here, and trigger an *update* on the delayed
   * sampling graph.
   *
   * - v: The random variate.
   * - future: The future value.
   */
  function assumeWithDowndate(v:Random<Value>, future:Value) {
    assert !v.hasDistribution();
    assert !v.hasValue();
    
    this.future <- future;
    futureUpdate <- false;
    v.p <- this;
  }
  
  /**
   * Observe a value for a random variate associated with this node,
   * updating (or downdating) the delayed sampling graph accordingly, and
   * returning a weight giving the log pdf (or pmf) of that variate under the
   * distribution.
   */
  function observe(x:Value) -> Real {
    assert !this.x?;
    assert !this.future?;
    prune();
    this.x <- x;
    this.futureUpdate <- true;
    return logpdf(x);
  }

  /**
   * Observe a value for a random variate associated with this node,
   * updating (or downdating) the delayed sampling graph accordingly, and
   * returning a weight giving the log pdf (or pmf) of that variate under the
   * distribution.
   */
  function observeWithDowndate(x:Value) -> Real {
    assert !this.x?;
    assert !this.future?;
    prune();
    this.x <- x;
    this.futureUpdate <- false;
    return logpdf(x);
  }

  function realize() {
    prune();
    if !x? {
      if future? {
        x <- future!;
      } else {
        x <- simulate();
      }
    }
    if futureUpdate {
      update(x!);
    } else {
      downdate(x!);
    }
  }
  
  /**
   * Simulate a value.
   *
   * Return: the value.
   */
  abstract function simulate() -> Value;

  /**
   * Simulate a pilot value.
   *
   * Return: the value.
   */
  function simulatePilot() -> Value {
    return simulate();
  }

  /**
   * Simulate a proposal value.
   *
   * Return: the value.
   */
  function simulatePropose() -> Value {
    return simulate();
  }

  /**
   * Log-pdf of a value.
   *
   * - x: The value.
   *
   * Return: The log likelihood.
   */
  abstract function logpdf(x:Value) -> Real;

  /**
   * Update the parent node on the $M$-path given the value of this node.
   *
   * - x: The value.
   */
  function update(x:Value) {
    //
  }

  /**
   * Downdate the parent node on the $M$-path given the value of this node.
   *
   * - x: The value.
   */
  function downdate(x:Value) {
    //
  }
  
  /**
   * Evaluate the probability density (or mass) function, if it exists.
   *
   * - x: The value.
   *
   * Return: the probability density (or mass).
   */
  function pdf(x:Value) -> Real {
    return exp(logpdf(x));
  }

  /**
   * Evaluate the cumulative distribution function at a value.
   *
   * - x: The value.
   *
   * Return: the cumulative probability, if supported.
   */
  function cdf(x:Value) -> Real? {
    return nil;
  }

  /**
   * Evaluate the quantile function at a cumulative probability.
   *
   * - P: The cumulative probability.
   *
   * Return: the quantile, if supported.
   */
  function quantile(P:Real) -> Value? {
    return nil;
  }
  
  /**
   * Finite lower bound of the support of this node, if any.
   */
  function lower() -> Value? {
    return nil;
  }
  
  /**
   * Finite upper bound of the support of this node, if any.
   */
  function upper() -> Value? {
    return nil;
  }

  /**
   * Graft this onto the delayed sampling graph.
   */
  function graft() -> Distribution<Value> {
    prune();
    return this;
  }

  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftGaussian() -> Gaussian? {
    return nil;
  }
    
  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftBeta() -> Beta? {
    return nil;
  }
  
  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftGamma() -> Gamma? {
    return nil;
  }
  
  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftInverseGamma() -> InverseGamma? {
    return nil;
  } 

  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftIndependentInverseGamma() -> IndependentInverseGamma? {
    return nil;
  } 

  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftInverseWishart() -> InverseWishart? {
    return nil;
  } 
  
  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftNormalInverseGamma() -> NormalInverseGamma? {
    return nil;
  }
  
  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftDirichlet() -> Dirichlet? {
    return nil;
  }

  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftRestaurant() -> Restaurant? {
    return nil;
  }

  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftMultivariateGaussian() -> MultivariateGaussian? {
    return nil;
  }

  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftMultivariateNormalInverseGamma() -> MultivariateNormalInverseGamma? {
    return nil;
  }

  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftMatrixGaussian() -> MatrixGaussian? {
    return nil;
  }

  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftMatrixNormalInverseGamma() -> MatrixNormalInverseGamma? {
    return nil;
  }

  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftMatrixNormalInverseWishart() -> MatrixNormalInverseWishart? {
    return nil;
  }

  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftDiscrete() -> Discrete? {
    return nil;
  }

  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftBoundedDiscrete() -> BoundedDiscrete? {
    return nil;
  }

  function write(buffer:Buffer) {
    buffer.set(value());
  }
}
