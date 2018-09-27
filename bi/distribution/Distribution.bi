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
   * Instantiate the associated delayed random variate by simulation.
   */
  function realize() {
    graft();
    delay!.realize();
    delay <- nil;
  }

  /**
   * Instantiate the associated delayed random variate by observation.
   */
  function realize(x:Value) -> Real {
    graft();
    w:Real <- delay!.realize(x);
    delay <- nil;
    return w;
  }

  /**
   * Simulate a random variate.
   *
   * Return: the value.
   */
  function simulate() -> Value {
    graft();
    x:Value <- delay!.simulate();
    delay!.condition(x);
    return x;
  }

  /**
   * Observe a random variate.
   *
   * - x: The value.
   *
   * Return: The log likelihood.
   */
  function observe(x:Value) -> Real {
    graft();
    w:Real <- delay!.observe(x);
    if (w > -inf) {
      delay!.condition(x);
    }
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
