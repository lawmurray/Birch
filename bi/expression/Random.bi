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

  function setChild(child:Delay) {
    assert x?;
  }

  function value() -> Value {
    if !x? {
      x <- dist!.value();
      dist <- nil;
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
    auto w <- dist!.observe(x);
    dist <- nil;
    return w;
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

  function graft() -> Expression<Value> {
    if !hasValue() {
      assert hasDistribution();
      dist <- dist!.graft();
      return DelayExpression<Value>(dist!);
    } else {
      return Boxed(x!);
    }
  }

  function graftGaussian() -> Gaussian? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftGaussian();
    } else {
      return nil;
    }
  }
    
  function graftBeta() -> Beta? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftBeta();
    } else {
      return nil;
    }
  }
  
  function graftGamma() -> Gamma? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftGamma();
    } else {
      return nil;
    }
  }
  
  function graftInverseGamma() -> InverseGamma? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftInverseGamma();
    } else {
      return nil;
    }
  } 

  function graftIndependentInverseGamma() -> IndependentInverseGamma? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftIndependentInverseGamma();
    } else {
      return nil;
    }
  } 

  function graftInverseWishart() -> InverseWishart? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftInverseWishart();
    } else {
      return nil;
    }
  } 
  
  function graftNormalInverseGamma() -> NormalInverseGamma? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftNormalInverseGamma();
    } else {
      return nil;
    }
  }
  
  function graftDirichlet() -> Dirichlet? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftDirichlet();
    } else {
      return nil;
    }
  }

  function graftRestaurant() -> Restaurant? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftRestaurant();
    } else {
      return nil;
    }
  }

  function graftMultivariateGaussian() -> MultivariateGaussian? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftMultivariateGaussian();
    } else {
      return nil;
    }
  }

  function graftMultivariateNormalInverseGamma() ->
      MultivariateNormalInverseGamma? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftMultivariateNormalInverseGamma();
    } else {
      return nil;
    }
  }

  function graftMatrixGaussian() -> MatrixGaussian? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftMatrixGaussian();
    } else {
      return nil;
    }
  }

  function graftMatrixNormalInverseGamma() -> MatrixNormalInverseGamma? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftMatrixNormalInverseGamma();
    } else {
      return nil;
    }
  }

  function graftMatrixNormalInverseWishart() -> MatrixNormalInverseWishart? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftMatrixNormalInverseWishart();
    } else {
      return nil;
    }
  }

  function graftDiscrete() -> Discrete? {
    if !hasValue() {
      assert hasDistribution();
      return dist!.graftDiscrete();
    } else {
      return nil;
    }
  }

  function graftBoundedDiscrete() -> BoundedDiscrete? {
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
