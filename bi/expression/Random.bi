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
   * Log-weight of prior.
   */
  w:Real <- 0.0;

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
    return x? || (p? && p!.hasValue());
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

  function distribution() -> Distribution<Value>? {
    return p;
  }

  /**
   * Assume the distribution for the random variate. When a value for the
   * random variate is required, it will be simulated from this distribution
   * and trigger an *update* on the delayed sampling graph.
   *
   * - p: The distribution.
   */
  function assume(p:Distribution<Value>) {
    assert !this.p?;
    assert !this.x?;
    this.p <- p;
  }

  function set(p:Distribution<Value>) {
    assume(p);
    this.x <- p.value();
  }

  function doValue() -> Value {
    graft();
    auto x <- p!.value();
    p <- nil;
    return x;
  }
  
  function doPilot() -> Value {
    graft();
    return p!.value();
  }

  function doGrad(d:Value) {
    assert x?;
    if p? {
      if dfdx? {
        y:Value <- dfdx! + d;
        dfdx <- y;
      } else {
        auto ψ <- p!.logpdfLazy(this);
        if ψ? {
          dfdx <- d;
          w <- ψ!.pilot();
          assert abs(w - p!.logpdf(x!)) < 1.0e-6;
          ψ!.grad(1.0);
        }
      }
    }
  }

  /**
   * Evaluate the log probability density (or mass) function, if it exists.
   *
   * - x: The value.
   *
   * Return: the log probability density (or mass).
   */
  function logpdf(x:Value) -> Real {
    graft();
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
    graft();
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
    graft();
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
    graft();
    return p!.quantile(P);
  }
  
  /**
   * Finite lower bound of the support of this node, if any.
   */
  function lower() -> Value? {
    graft();
    return p!.lower();
  }
  
  /**
   * Finite upper bound of the support of this node, if any.
   */
  function upper() -> Value? {
    graft();
    return p!.upper();
  }

  function graft() {
    if !hasValue() {
      p <- p!.graft();
    }
  }

  function graftGaussian() -> Gaussian? {
    if !hasValue() {
      auto q <- p!.graftGaussian();
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  }
    
  function graftBeta() -> Beta? {
    if !hasValue() {
      auto q <- p!.graftBeta();
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  }
  
  function graftGamma() -> Gamma? {
    if !hasValue() {
      auto q <- p!.graftGamma();
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  }
  
  function graftInverseGamma() -> InverseGamma? {
    if !hasValue() {
      auto q <- p!.graftInverseGamma();
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  } 

  function graftIndependentInverseGamma() -> IndependentInverseGamma? {
    if !hasValue() {
      auto q <- p!.graftIndependentInverseGamma();
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  } 

  function graftInverseWishart() -> InverseWishart? {
    if !hasValue() {
      auto q <- p!.graftInverseWishart();
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  } 
  
  function graftNormalInverseGamma(compare:Distribution<Real>) -> NormalInverseGamma? {
    if !hasValue() {
      auto q <- p!.graftNormalInverseGamma(compare);
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  }
  
  function graftDirichlet() -> Dirichlet? {
    if !hasValue() {
      auto q <- p!.graftDirichlet();
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  }

  function graftRestaurant() -> Restaurant? {
    if !hasValue() {
      auto q <- p!.graftRestaurant();
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  }

  function graftMultivariateGaussian() -> MultivariateGaussian? {
    if !hasValue() {
      auto q <- p!.graftMultivariateGaussian();
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  }

  function graftMultivariateNormalInverseGamma(compare:Distribution<Real>) ->
      MultivariateNormalInverseGamma? {
    if !hasValue() {
      auto q <- p!.graftMultivariateNormalInverseGamma(compare);
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  }

  function graftMatrixGaussian() -> MatrixGaussian? {
    if !hasValue() {
      auto q <- p!.graftMatrixGaussian();
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  }

  function graftMatrixNormalInverseGamma(compare:Distribution<Real[_]>) ->
      MatrixNormalInverseGamma? {
    if !hasValue() {
      auto q <- p!.graftMatrixNormalInverseGamma(compare);
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  }

  function graftMatrixNormalInverseWishart(compare:Distribution<Real[_,_]>) ->
      MatrixNormalInverseWishart? {
    if !hasValue() {
      auto q <- p!.graftMatrixNormalInverseWishart(compare);
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  }

  function graftDiscrete() -> Discrete? {
    if !hasValue() {
      auto q <- p!.graftDiscrete();
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  }

  function graftBoundedDiscrete() -> BoundedDiscrete? {
    if !hasValue() {
      auto q <- p!.graftBoundedDiscrete();
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
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
