/**
 * Random variate.
 *
 * - Value: Value type.
 */
final class Random<Value> < Expression<Value> {  
  /**
   * Associated distribution.
   */
  p:Distribution<Value>?;

  /**
   * Final value.
   */
  x:Value?;

  /**
   * Value assignment.
   */
  operator <- x:Value {
    assert !p?;
    this.x <- x;
  }

  /**
   * Optional value assignment.
   */
  operator <- x:Value? {
    assert !p?;
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
    return p?;
  }
  
  function rows() -> Integer {
    if x? {
      return global.rows(x!);
    } else {
      assert p?;
      return p!.rows();
    }
  }

  function columns() -> Integer {
    if x? {
      return global.columns(x!);
    } else {
      assert p?;
      return p!.columns();
    }
  }

  function setChild(child:Delay) {
    assert x?;
  }

  function value() -> Value {
    if !x? {
      x <- p!.value();
      p <- nil;
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
    assert p?;
    this.x <- x;
    auto w <- p!.observe(x);
    p <- nil;
    return w;
  }
  
  /**
   * Get the distribution associated with the random variate.
   */
  function distribution() -> Distribution<Value> {
    assert p?;
    return p!;
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
    return p!.logpdf(x);
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
    return p!.pdf(x);
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
    return p!.cdf(x);
  }

  /**
   * Evaluate the quantile function at a cumulative probability.
   *
   * - P: The cumulative probability.
   *
   * Return: the quantile value, if supported.
   */
  function quantile(P:Real) -> Value? {
    assert hasDistribution();
    return p!.quantile(P);
  }
  
  /**
   * Finite lower bound of the support of this node, if any.
   */
  function lower() -> Value? {
    assert hasDistribution();
    return p!.lower();
  }
  
  /**
   * Finite upper bound of the support of this node, if any.
   */
  function upper() -> Value? {
    assert hasDistribution();
    return p!.upper();
  }

  function graft() -> Expression<Value> {
    if !hasValue() {
      assert hasDistribution();
      p <- p!.graft();
      return DelayExpression<Value>(p!);
    } else {
      return Boxed(x!);
    }
  }

  function graftGaussian() -> Gaussian? {
    if !hasValue() {
      assert hasDistribution();
      return p!.graftGaussian();
    } else {
      return nil;
    }
  }
    
  function graftBeta() -> Beta? {
    if !hasValue() {
      assert hasDistribution();
      return p!.graftBeta();
    } else {
      return nil;
    }
  }
  
  function graftGamma() -> Gamma? {
    if !hasValue() {
      assert hasDistribution();
      return p!.graftGamma();
    } else {
      return nil;
    }
  }
  
  function graftInverseGamma() -> InverseGamma? {
    if !hasValue() {
      assert hasDistribution();
      return p!.graftInverseGamma();
    } else {
      return nil;
    }
  } 

  function graftIndependentInverseGamma() -> IndependentInverseGamma? {
    if !hasValue() {
      assert hasDistribution();
      return p!.graftIndependentInverseGamma();
    } else {
      return nil;
    }
  } 

  function graftInverseWishart() -> InverseWishart? {
    if !hasValue() {
      assert hasDistribution();
      return p!.graftInverseWishart();
    } else {
      return nil;
    }
  } 
  
  function graftNormalInverseGamma() -> NormalInverseGamma? {
    if !hasValue() {
      assert hasDistribution();
      return p!.graftNormalInverseGamma();
    } else {
      return nil;
    }
  }
  
  function graftDirichlet() -> Dirichlet? {
    if !hasValue() {
      assert hasDistribution();
      return p!.graftDirichlet();
    } else {
      return nil;
    }
  }

  function graftRestaurant() -> Restaurant? {
    if !hasValue() {
      assert hasDistribution();
      return p!.graftRestaurant();
    } else {
      return nil;
    }
  }

  function graftMultivariateGaussian() -> MultivariateGaussian? {
    if !hasValue() {
      assert hasDistribution();
      return p!.graftMultivariateGaussian();
    } else {
      return nil;
    }
  }

  function graftMultivariateNormalInverseGamma() ->
      MultivariateNormalInverseGamma? {
    if !hasValue() {
      assert hasDistribution();
      return p!.graftMultivariateNormalInverseGamma();
    } else {
      return nil;
    }
  }

  function graftMatrixGaussian() -> MatrixGaussian? {
    if !hasValue() {
      assert hasDistribution();
      return p!.graftMatrixGaussian();
    } else {
      return nil;
    }
  }

  function graftMatrixNormalInverseGamma() -> MatrixNormalInverseGamma? {
    if !hasValue() {
      assert hasDistribution();
      return p!.graftMatrixNormalInverseGamma();
    } else {
      return nil;
    }
  }

  function graftMatrixNormalInverseWishart() -> MatrixNormalInverseWishart? {
    if !hasValue() {
      assert hasDistribution();
      return p!.graftMatrixNormalInverseWishart();
    } else {
      return nil;
    }
  }

  function graftDiscrete() -> Discrete? {
    if !hasValue() {
      assert hasDistribution();
      return p!.graftDiscrete();
    } else {
      return nil;
    }
  }

  function graftBoundedDiscrete() -> BoundedDiscrete? {
    if !hasValue() {
      assert hasDistribution();
      return p!.graftBoundedDiscrete();
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
      p!.write(buffer);
    } else {
      buffer.setNil();
    }
  }
}
