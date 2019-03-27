
/**
 * Random variate.
 *
 * - Value: Value type.
 */
class Random<Value> < Expression<Value> {  
  /**
   * Value.
   */
  x:Value?;
  
  /**
   * Future value.
   */
  future:Value?;

  /**
   * Associated distribution.
   */
  dist:Distribution<Value>?;

  /**
   * Value assignment.
   */
  operator <- x:Value {
    assert !hasDistribution();
    assert !hasValue();
    this.x <- x;
  }

  /**
   * Optional value assignment.
   */
  operator <- x:Value? {
    assert !hasDistribution();
    assert !hasValue();
    this.x <- x;
  }

  /**
   * Attach a distribution to this random variate.
   */
  function assume(dist:Distribution<Value>) {
    assert !hasDistribution();
    assert !hasValue();
    
    dist.associate(this);
    this.dist <- dist;
  }

  /**
   * Attach a distribution to this random variate, and a future value.
   *
   * - dist: The distribution.
   * - future: The future value.
   *
   * The random variate is treated as though it has no value. When a value
   * must be realized, however, that future value given here will be used. No
   * weight adjustment is made for this. This function is mostly provided for
   * the purposes of replaying model executions, using e.g. ReplayHandler.
   */
  function assume(dist:Distribution<Value>, future:Value) {
    assume(dist);
    this.future <- future;
  }

  /**
   * Get the value of the random variate, forcing realization if necessary.
   */
  function value() -> Value {
    if !hasValue() {
      assert hasDistribution();
      if future? {
        /* future value was provided, use it */
        dist!.realize(future!);
        future <- nil;
      } else {
        dist!.realize();
      }
      dist <- nil;
      assert hasValue();
    }
    return x!;
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

  /**
   * Evaluate the probability mass function (if it exists) at a value.
   *
   * - x: The value.
   *
   * Return: the probability mass.
   */
  function pmf(x:Value) -> Real {
    assert hasDistribution();
    return dist!.pmf(x);
  }

  /**
   * Evaluate the probability density function (if it exists) at a value.
   *
   * - x: The value.
   *
   * Return: the probability density.
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
   * Return: the cumulative probability
   */
  function cdf(x:Value) -> Real {
    assert hasDistribution();
    return dist!.cdf(x);
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

  function hasDelay() -> Boolean {
    if !hasValue() {
      assert hasDistribution();
      return dist!.delay?;
    } else {
      return false;
    }
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
  
  function graftNormalInverseGamma(σ2:Expression<Real>) ->
      DelayNormalInverseGamma? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftNormalInverseGamma(σ2);
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

  function graftMultivariateNormalInverseGamma(σ2:Expression<Real>) ->
      DelayMultivariateNormalInverseGamma? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftMultivariateNormalInverseGamma(σ2);
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
    if hasValue() || hasDistribution() {
      buffer.set(value());
    } else {
      buffer.setNil();
    }
  }
}

/**
 * Create random variate with attached distribution.
 */
function Random(dist:Distribution<Boolean>) -> Random<Boolean> {
  m:Random<Boolean>;
  m.assume(dist);
  return m;
}

/**
 * Create random variate with attached distribution.
 */
function Random(dist:Distribution<Integer>) -> Random<Integer> {
  m:Random<Integer>;
  m.assume(dist);
  return m;
}

/**
 * Create random variate with attached distribution.
 */
function Random(dist:Distribution<Real>) -> Random<Real> {
  m:Random<Real>;
  m.assume(dist);
  return m;
}

/**
 * Create random variate with attached distribution.
 */
function Random(dist:Distribution<Boolean[_]>) -> Random<Boolean[_]> {
  m:Random<Boolean[_]>;
  m.assume(dist);
  return m;
}

/**
 * Create random variate with attached distribution.
 */
function Random(dist:Distribution<Integer[_]>) -> Random<Integer[_]> {
  m:Random<Integer[_]>;
  m.assume(dist);
  return m;
}

/**
 * Create random variate with attached distribution.
 */
function Random(dist:Distribution<Real[_]>) -> Random<Real[_]> {
  m:Random<Real[_]>;
  m.assume(dist);
  return m;
}
