/**
 * Probability distribution.
 *
 * - Value: Value type.
 */
class Distribution<Value> {
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

  /**
   * Set the future value to be produced from the distribution.
   *
   * - future: Future value.
   * - futureUpdate: When realized, should the future value trigger an
   *   update? (Otherwise a downdate.)
   */
  function setFuture(future:Value, futureUpdate:Boolean) {
    this.future <- future;
    this.futureUpdate <- futureUpdate;
  }

  /**
   * Does the distribution have a value?
   */
  function hasValue() -> Boolean {
    return delay? && delay!.hasValue();
  }
  
  /**
   * Get the value of the node, realizing it if necessary.
   */
  function value() -> Value {
    graft();
    auto x <- delay!.value();
    detach();
    return x;
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
   * Observe a random variate.
   *
   * - x: The observed value.
   *
   * Return: The log likelihood.
   */
  function observe(x:Value) -> Real {
    graft();
    return delay!.observe(x);
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
   * Evaluate the probability mass function (if it exists) at a value.
   *
   * - x: The value.
   *
   * Return: the probability mass.
   */
  function pmf(x:Value) -> Real {
    graft();
    return delay!.pmf(x);
  }

  /**
   * Evaluate the probability density function (if it exists) at a value.
   *
   * - x: The value.
   *
   * Return: the probability density.
   */
  function pdf(x:Value) -> Real {
    graft();
    return delay!.pdf(x);
  }

  /**
   * Evaluate the cumulative distribution function at a value.
   *
   * - x: The value.
   *
   * Return: the cumulative probability
   */
  function cdf(x:Value) -> Real {
    graft();
    return delay!.cdf(x);
  }
  
  /**
   * Finite lower bound of the support of this node, if any.
   */
  function lower() -> Value? {
    graft();
    return delay!.lower();
  }
  
  /**
   * Finite upper bound of the support of this node, if any.
   */
  function upper() -> Value? {
    graft();
    return delay!.upper();
  }
  
  /**
   * Graft this onto the delayed sampling $M$-path.
   */
  function graft();
  
  /**
   * Detach this from the delayed sampling $M$-path.
   */
  function detach() {
    assert delay?;
    delay!.detach();
    delay <- nil;
  }

  function graftGaussian() -> DelayGaussian? {
    return nil;
  }

  function graftRidge() -> DelayRidge? {
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
  
  function graftNormalInverseGamma(σ2:Expression<Real>) ->
      DelayNormalInverseGamma? {
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

  function graftMultivariateNormalInverseGamma(σ2:Expression<Real>) ->
      DelayMultivariateNormalInverseGamma? {
    return nil;
  }

  function graftDiscrete() -> DelayDiscrete? {
    return nil;
  }

  function graftBoundedDiscrete() -> DelayBoundedDiscrete? {
    return nil;
  }
}
