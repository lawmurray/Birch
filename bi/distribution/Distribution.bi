/**
 * Probability distribution.
 *
 * - Value: Value type.
 */
abstract class Distribution<Value> {
  /**
   * Future value. This is set for situations where delayed sampling
   * is used, but when ultimately realized, a particular value (this one)
   * should be assigned, and updates or downdates applied accordingly. It
   * is typically used when replaying traces.
   */
  future:Value?;

  /**
   * When assigned, should the future value trigger an update? (Otherwise
   * a downdate.)
   */
  futureUpdate:Boolean <- true;

  /**
   * Associated node on delayed sampling $M$-path.
   */
  delay:DelayValue<Value>?;

  function value() -> Value {
    graft();
    return delay!.value();
  }

  function pilot() -> Value {
    assert delay?;
    return delay!.pilot();
  }
  
  function propose() -> Value {
    assert delay?;
    return delay!.propose();
  }
  
  function set(x:Value) {
    assert delay?;
    delay!.set(x);
  }

  function rows() -> Integer {
    return 1;
  }
  
  function columns() -> Integer {
    return 1;
  }

  /**
   * Assume a distribution for this random variate. When a value is required,
   * it will be simulated from this distribution and trigger an update on
   * the delayed sampling graph.
   */
  function assume(v:Random<Value>) {
    assert !v.hasDistribution();
    assert !v.hasValue();
    
    futureUpdate <- true;
    v.dist <- this;
  }

  /**
   * Assume a distribution for this random variate. When a value is required,
   * it will be assigned according to the `future` value given here, and
   * trigger an update on the delayed sampling graph.
   *
   * - dist: The distribution.
   * - future: The future value.
   */
  function assume(v:Random<Value>, future:Value) {
    assert !v.hasDistribution();
    assert !v.hasValue();
    
    this.future <- future;
    futureUpdate <- true;
    v.dist <- this;
  }

  /**
   * Assume a distribution for this random variate. When a value is required,
   * it will be simulated from this distribution and trigger a downdate on
   * the delayed sampling graph.
   */
  function assumeWithDowndate(v:Random<Value>) {
    assert !v.hasDistribution();
    assert !v.hasValue();
    
    futureUpdate <- false;
    v.dist <- this;
  }

  /**
   * Assume a distribution for this random variate. When a value is required,
   * it will be assigned according to the `future` value given here, and
   * trigger a downdate on the delayed sampling graph.
   *
   * - dist: The distribution.
   * - future: The future value.
   */
  function assumeWithDowndate(v:Random<Value>, future:Value) {
    assert !v.hasDistribution();
    assert !v.hasValue();
    
    this.future <- future;
    futureUpdate <- false;
    v.dist <- this;
  }
  
  /**
   * Observe a value for a random variate associated with the distribution,
   * updating the delayed sampling graph accordingly, and returning a weight
   * giving the log pdf (or pmf) of that variate under the distribution.
   */
  function observe(x:Value) -> Real {
    graft();
    auto w <- delay!.observe(x);
    return w;
  }

  /**
   * Observe a value for a random variate associated with the distribution,
   * downdating the delayed sampling graph accordingly, and returning a weight
   * giving the log pdf (or pmf) of that variate under the distribution.
   */
  function observeWithDowndate(x:Value) -> Real {
    graft();
    auto w <- delay!.observeWithDowndate(x);
    return w;
  }

  /**
   * Simulate a random variate.
   *
   * Return: The simulated value.
   */
  function simulate() -> Value {
    graft();
    return delay!.simulate();
  }

  /**
   * Update the parameters of the distribution with a given value.
   *
   * Return: The value.
   */
  function update(x:Value) {
    graft();
    delay!.update(x);
  }

  /**
   * Downdate the parameters of the distribution with a given value. This
   * undoes the effects of an update().
   *
   * - x: The value.
   */
  function downdate(x:Value) {
    graft();
    delay!.downdate(x);
  }

  /**
   * Evaluate the logarithm of the probability density (or mass) function.
   *
   * - x: The value.
   *
   * Return: the log probability density (or mass).
   */
  function logpdf(x:Value) -> Real {
    graft();
    return delay!.logpdf(x);
  }

  /**
   * Evaluate the probability density (or mass) function, if it exists.
   *
   * - x: The value.
   *
   * Return: the probability density (or mass).
   */
  function pdf(x:Value) -> Real {
    graft();
    return delay!.pdf(x);
  }

  /**
   * Evaluate the cumulative distribution function at a value, if supported.
   *
   * - x: The value.
   *
   * Return: the cumulative probability.
   */
  function cdf(x:Value) -> Real? {
    graft();
    return delay!.cdf(x);
  }

  /**
   * Evaluate the quantile function at a cumulative probability, if supported.
   *
   * - x: The cumulative probability.
   *
   * Return: the quantile.
   */
  function quantile(p:Real) -> Value? {
    graft();
    return delay!.quantile(p);
  }
  
  /**
   * Finite lower bound of the support of this node, if supported.
   */
  function lower() -> Value? {
    graft();
    return delay!.lower();
  }
  
  /**
   * Finite upper bound of the support of this node, if supported.
   */
  function upper() -> Value? {
    graft();
    return delay!.upper();
  }
  
  /**
   * Graft this onto the delayed sampling graph.
   *
   * - force: If true, a node is always grafted onto the delayed sampling
   *   graph, even if it has no parent. If false, no node is grafted in this
   *   case.
   */
  abstract function graft();

  function graftGaussian() -> DelayGaussian? {
    return nil;
  }
    
  function graftBeta() -> DelayBeta? {
    return nil;
  }
  
  function graftGamma() -> DelayGamma? {
    return nil;
  }
  
  function graftInverseGamma() -> DelayInverseGamma? {
    return nil;
  } 

  function graftIndependentInverseGamma() -> DelayIndependentInverseGamma? {
    return nil;
  } 

  function graftInverseWishart() -> DelayInverseWishart? {
    return nil;
  } 
  
  function graftNormalInverseGamma() -> DelayNormalInverseGamma? {
    return nil;
  }
  
  function graftDirichlet() -> DelayDirichlet? {
    return nil;
  }

  function graftRestaurant() -> DelayRestaurant? {
    return nil;
  }

  function graftMultivariateGaussian() -> DelayMultivariateGaussian? {
    return nil;
  }

  function graftMultivariateNormalInverseGamma() -> DelayMultivariateNormalInverseGamma? {
    return nil;
  }

  function graftMatrixGaussian() -> DelayMatrixGaussian? {
    return nil;
  }

  function graftMatrixNormalInverseGamma() -> DelayMatrixNormalInverseGamma? {
    return nil;
  }

  function graftMatrixNormalInverseWishart() -> DelayMatrixNormalInverseWishart? {
    return nil;
  }

  function graftDiscrete() -> DelayDiscrete? {
    return nil;
  }

  function graftBoundedDiscrete() -> DelayBoundedDiscrete? {
    return nil;
  }

  function write(buffer:Buffer) {
    graft();
    delay!.write(buffer);
  }
}
