/**
 * Random variate.
 *
 * - Value: Value type.
 */
final class Random<Value> < Expression<Value> {  
  /**
   * Realized value.
   */
  x:Value?;
  
  /**
   * Pilot value.
   */
  xstar:Value?;
  
  /**
   * Propose value.
   */
  xprime:Value?;
  
  /**
   * Gradient a function in which this object occurs, at the pilot position.
   */
  dstar:Value?;

  /**
   * Gradient a function in which this object occurs, at the propose position.
   */
  dprime:Value?;

  /**
   * Associated distribution.
   */
  dist:Distribution<Value>?;

  /**
   * Value assignment.
   */
  operator <- x:Value {
    assert !hasDistribution();
    this.x <- x;
  }

  /**
   * Optional value assignment.
   */
  operator <- x:Value? {
    assert !hasDistribution();
    this.x <- x;
  }

  /**
   * Does this have a value?
   */
  function hasValue() -> Boolean {
    return x?;
  }

  /**
   * Does this have a distribution?
   */
  function hasDistribution() -> Boolean {
    return dist?;
  }

  function value() -> Value {
    if !x? {
      assert dist?;
      x <- dist!.value();
      dist <- nil;
    }
    return x!;
  }

  function pilot() -> Value {
    if x? {
      return x!;
    } else if !xstar? {
      assert dist?;
      xstar <- dist!.simulate();
    }
    return xstar!;
  }

  function propose() -> Value {
    if x? {
      return x!;
    } else if !xprime? {
      assert dist?;
      xprime <- simulate_propose(dstar!, dstar!);
    }
    return xprime!;
  }
  
  function dpilot(d:Value) {
    if this.dstar? {
      this.dstar! <- this.dstar! + d;
    } else {
      this.dstar <- d;
    }
  }

  function dpropose(d:Value) {
    if this.dprime? {
      this.dprime! <- this.dprime! + d;
    } else {
      this.dprime <- d;
    }
  }

  function ratio() -> Real {
    return 0.0;
  }
  
  function accept() {
    x <- xprime;
    xstar <- nil;
    xprime <- nil;
    dstar <- nil;
    dprime <- nil;
    dist <- nil;
  }

  function reject() {
    x <- xstar;
    xstar <- nil;
    xprime <- nil;
    dstar <- nil;
    dprime <- nil;
    dist <- nil;
  }

  /**
   * Observe the value of the random variate.
   */
  function observe(x:Value) -> Real {
    assert !this.x?;
    assert dist?;
    this.x <- x;
    return dist!.observe(x);
  }
  
  /**
   * Get the distribution associated with the random variate.
   */
  function distribution() -> Distribution<Value> {
    assert dist?;
    return dist!;
  }

  /**
   * Evaluate the log probability density (or mass) function, if it exists.
   *
   * - x: The value.
   *
   * Return: the log probability density (or mass).
   */
  function logpdf(x:Value) -> Real {
    assert hasDistribution();
    return dist!.logpdf(x);
  }

  /**
   * Evaluate the probability density (or mass) function, if it exists.
   *
   * - x: The value.
   *
   * Return: the probability density (or mass).
   */
  function pdf(x:Value) -> Real {
    assert hasDistribution();
    return dist!.pdf(x);
  }

  /**
   * Evaluate the cumulative distribution function at a value.
   *
   * - x: The value.
   *
   * Return: the cumulative probability, if supported.
   */
  function cdf(x:Value) -> Real? {
    assert hasDistribution();
    return dist!.cdf(x);
  }

  /**
   * Evaluate the quantile function at a cumulative probability.
   *
   * - x: The cumulative probability.
   *
   * Return: the quantile value, if supported.
   */
  function quantile(p:Real) -> Value? {
    assert hasDistribution();
    return dist!.quantile(p);
  }
  
  /**
   * Finite lower bound of the support of this node, if any.
   */
  function lower() -> Value? {
    assert hasDistribution();
    return dist!.lower();
  }
  
  /**
   * Finite upper bound of the support of this node, if any.
   */
  function upper() -> Value? {
    assert hasDistribution();
    return dist!.upper();
  }

  function getDelay() -> Delay? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.delay;
    } else {
      return nil;
    }
  }

  function graftGaussian() -> DelayGaussian? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftGaussian();
    } else {
      return nil;
    }
  }
    
  function graftBeta() -> DelayBeta? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftBeta();
    } else {
      return nil;
    }
  }
  
  function graftGamma() -> DelayGamma? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftGamma();
    } else {
      return nil;
    }
  }
  
  function graftInverseGamma() -> DelayInverseGamma? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftInverseGamma();
    } else {
      return nil;
    }
  } 

  function graftIndependentInverseGamma() -> DelayIndependentInverseGamma? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftIndependentInverseGamma();
    } else {
      return nil;
    }
  } 

  function graftInverseWishart() -> DelayInverseWishart? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftInverseWishart();
    } else {
      return nil;
    }
  } 
  
  function graftNormalInverseGamma() -> DelayNormalInverseGamma? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftNormalInverseGamma();
    } else {
      return nil;
    }
  }
  
  function graftDirichlet() -> DelayDirichlet? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftDirichlet();
    } else {
      return nil;
    }
  }

  function graftRestaurant() -> DelayRestaurant? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftRestaurant();
    } else {
      return nil;
    }
  }

  function graftMultivariateGaussian() -> DelayMultivariateGaussian? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftMultivariateGaussian();
    } else {
      return nil;
    }
  }

  function graftMultivariateNormalInverseGamma() ->
      DelayMultivariateNormalInverseGamma? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftMultivariateNormalInverseGamma();
    } else {
      return nil;
    }
  }

  function graftMatrixGaussian() -> DelayMatrixGaussian? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftMatrixGaussian();
    } else {
      return nil;
    }
  }

  function graftMatrixNormalInverseGamma() -> DelayMatrixNormalInverseGamma? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftMatrixNormalInverseGamma();
    } else {
      return nil;
    }
  }

  function graftMatrixNormalInverseWishart() -> DelayMatrixNormalInverseWishart? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftMatrixNormalInverseWishart();
    } else {
      return nil;
    }
  }

  function graftDiscrete() -> DelayDiscrete? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftDiscrete();
    } else {
      return nil;
    }
  }

  function graftBoundedDiscrete() -> DelayBoundedDiscrete? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftBoundedDiscrete();
    } else {
      return nil;
    }
  }

  function read(buffer:Buffer) {
    assert !hasDistribution();
    assert !hasValue();
    x <- buffer.get(x);
  }

  function write(buffer:Buffer) {
    if hasValue() {
      buffer.set(value());
    } else if hasDistribution() {
      dist!.write(buffer);
    } else {
      buffer.setNil();
    }
  }
}

function simulate_propose(x:Real, d:Real) -> Real {
  return simulate_gaussian(x + d, 1.0);
}

function simulate_propose(x:Real[_], d:Real[_]) -> Real[_] {
  return simulate_multivariate_gaussian(x + d, 1.0);
}

function simulate_propose(x:Real[_,_], d:Real[_,_]) -> Real[_,_] {
  return simulate_matrix_gaussian(x + d, 1.0);
}

function simulate_propose(x:Integer, d:Integer) -> Integer {
  return x;
}

function simulate_propose(x:Integer[_], d:Integer[_]) -> Integer[_] {
  return x;
}

function simulate_propose(x:Boolean, d:Boolean) -> Boolean {
  return x;
}
