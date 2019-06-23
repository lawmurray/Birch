
/**
 * Random variate.
 *
 * - Value: Value type.
 */
final class Random<Value> < Expression<Value> {  
  /**
   * Value.
   */
  x:Value?;

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
   * Get the value of the random variate, forcing realization if necessary.
   */
  function value() -> Value {
    if !x? {
      assert dist?;
      x <- dist!.value();
      dist <- nil;
    }
    return x!;
  }
  
  /**
   * Set the value of the random variate, returning a weight giving the log
   * pdf (or pmf) of that variate under the assumed distribution.
   */
  function observe(x:Value) -> Real {
    assert !this.x?;
    assert dist?;
    this.x <- x;
    return dist!.observe(x);
  }

  /**
   * Assume a distribution for this random variate. When a value is required,
   * it will be simuldated from this distribution.
   */
  function assume(dist:Distribution<Value>) {
    assert !hasDistribution();
    assert !hasValue();
    this.dist <- dist;
  }

  /**
   * Assume a distribution for this random variate. When a value is required,
   * it will be assigned according to the `future` value given here, and
   * trigger an update on the delayed sampling graph.
   *
   * - dist: The distribution.
   * - future: The future value.
   */
  function assumeUpdate(dist:Distribution<Value>, future:Value) {
    dist.setFuture(future, true);
    assume(dist);
  }

  /**
   * Assume a distribution for this random variate. When a value is required,
   * it will be assigned according to the `future` value given here, and
   * trigger a downdate on the delayed sampling graph.
   *
   * - dist: The distribution.
   * - future: The future value.
   */
  function assumeDowndate(dist:Distribution<Value>, future:Value) {
    dist.setFuture(future, false);
    assume(dist);
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

  function graftRidge() -> DelayRidge? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftRidge();
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
    if hasValue() {
      buffer.set(value());
    } else if hasDistribution() {
      dist!.write(buffer);
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
