/**
 * Random variate.
 *
 * - Value: Value type.
 */
final class Random<Value> < Expression<Value> {  
  /**
   * Associated distribution.
   */
  dist:Distribution<Value>?;

  /**
   * Final value.
   */
  x:Value?;

  /**
   * Value assignment.
   */
  operator <- x:Value {
    assert !dist?;
    this.x <- x;
  }

  /**
   * Optional value assignment.
   */
  operator <- x:Value? {
    assert !dist?;
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
  
  function rows() -> Integer {
    if x? {
      return global.rows(x!);
    } else {
      assert dist?;
      return dist!.rows();
    }
  }

  function columns() -> Integer {
    if x? {
      return global.columns(x!);
    } else {
      assert dist?;
      return dist!.columns();
    }
  }

  function value() -> Value {
    if !x? {
      x <- dist!.value();
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
    assert x?;
    return false;
  }

  function gradPropose(d:Value) -> Boolean {
    assert x?;
    return false;
  }
  
  function ratio() -> Real {
    assert x?;
    return 0.0;
  }
  
  function accept() {
    assert x?;
  }

  function reject() {
    assert x?;
  }

  function clamp() {
    assert x?;
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

  function graft(child:Delay?) -> Expression<Value> {
    if x? {
      return Boxed(x!);
    } else {
      dist!.graft(child);
      return DelayExpression<Value>(dist!.delay!);
    }
  }

  function graftGaussian(child:Delay?) -> DelayGaussian? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftGaussian(child);
    } else {
      return nil;
    }
  }
    
  function graftBeta(child:Delay?) -> DelayBeta? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftBeta(child);
    } else {
      return nil;
    }
  }
  
  function graftGamma(child:Delay?) -> DelayGamma? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftGamma(child);
    } else {
      return nil;
    }
  }
  
  function graftInverseGamma(child:Delay?) -> DelayInverseGamma? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftInverseGamma(child);
    } else {
      return nil;
    }
  } 

  function graftIndependentInverseGamma(child:Delay?) -> DelayIndependentInverseGamma? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftIndependentInverseGamma(child);
    } else {
      return nil;
    }
  } 

  function graftInverseWishart(child:Delay?) -> DelayInverseWishart? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftInverseWishart(child);
    } else {
      return nil;
    }
  } 
  
  function graftNormalInverseGamma(child:Delay?) -> DelayNormalInverseGamma? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftNormalInverseGamma(child);
    } else {
      return nil;
    }
  }
  
  function graftDirichlet(child:Delay?) -> DelayDirichlet? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftDirichlet(child);
    } else {
      return nil;
    }
  }

  function graftRestaurant(child:Delay?) -> DelayRestaurant? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftRestaurant(child);
    } else {
      return nil;
    }
  }

  function graftMultivariateGaussian(child:Delay?) -> DelayMultivariateGaussian? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftMultivariateGaussian(child);
    } else {
      return nil;
    }
  }

  function graftMultivariateNormalInverseGamma(child:Delay?) ->
      DelayMultivariateNormalInverseGamma? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftMultivariateNormalInverseGamma(child);
    } else {
      return nil;
    }
  }

  function graftMatrixGaussian(child:Delay?) -> DelayMatrixGaussian? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftMatrixGaussian(child);
    } else {
      return nil;
    }
  }

  function graftMatrixNormalInverseGamma(child:Delay?) -> DelayMatrixNormalInverseGamma? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftMatrixNormalInverseGamma(child);
    } else {
      return nil;
    }
  }

  function graftMatrixNormalInverseWishart(child:Delay?) -> DelayMatrixNormalInverseWishart? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftMatrixNormalInverseWishart(child);
    } else {
      return nil;
    }
  }

  function graftDiscrete(child:Delay?) -> DelayDiscrete? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftDiscrete(child);
    } else {
      return nil;
    }
  }

  function graftBoundedDiscrete(child:Delay?) -> DelayBoundedDiscrete? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftBoundedDiscrete(child);
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
  return simulate_gaussian(x + 0.04*d, 0.08);
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

function logpdf_propose(x':Real, x:Real, d:Real) -> Real {
  return logpdf_gaussian(x', x + 0.04*d, 0.08);
}

function logpdf_propose(x':Real[_], x:Real[_], d:Real[_]) -> Real {
  return logpdf_multivariate_gaussian(x', x + d, 1.0);
}

function logpdf_propose(x':Real[_,_], x:Real[_,_], d:Real[_,_]) -> Real {
  return logpdf_matrix_gaussian(x', x + d, 1.0);
}

function logpdf_propose(x':Integer, x:Integer, d:Integer) -> Real {
  return 0.0;
}

function logpdf_propose(x':Integer[_], x:Integer[_], d:Integer[_]) -> Real {
  return 0.0;
}

function logpdf_propose(x':Boolean, x:Boolean, d:Boolean) -> Real {
  return 0.0;
}
