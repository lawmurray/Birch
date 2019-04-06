/**
 * Probability distribution.
 *
 * - Value: Value type.
 */
class Distribution<Value> {
  /**
   * Associated node on delayed sampling $M$-path.
   */
  delay:DelayValue<Value>?;

  /**
   * Associated random variate.
   */
  x:Random<Value>&;

  /**
   * Associate a random variate with this distribution.
   */
  function associate(x:Random<Value>) {
    this.x <- x;
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
   * Simulate then update.
   *
   * Return: The simulated value.
   */
  function simulateAndUpdate() -> Value {
    auto x <- simulate();
    update(x);
    return x;
  }

  /**
   * Observe then update.
   *
   * - x: The observed value.
   *
   * Return: The log likelihood.
   */
  function observeAndUpdate(x:Value) -> Real {
    auto w <- observe(x);
    update(x);
    return w;
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
    delay!.detach();
    delay <- nil;
  }

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
