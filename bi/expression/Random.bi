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
   * Value.
   */
  x:Value?;
  
  /**
   * Gradient.
   */
  dfdx:Value?;

  /**
   * Log-weight of prior.
   */
  w:Real?;

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
    assert hasValue();
  }

  function distribution() -> Distribution<Value>? {
    return p;
  }

  function value() -> Value {
    if !x? {
      p <- p!.graft();
      x <- p!.value();
      p <- nil;
    }
    return x!;
  }

  function grad(d:Value) {
    assert x?;
    if p? {
      if dfdx? {
        dfdx <- dfdx! + d;
      } else {
        dfdx <- d;
        auto ψ <- p!.lazy(this);
        if ψ? {
          w <- ψ!.value();
          ψ!.grad(1.0);
        }
      }
    }
  }

  /**
   * Observe the value of the random variate.
   */
  function observe(x:Value) -> Real {
    assert !hasValue();
    assert hasDistribution();
    this.x <- x;
    p <- p!.graft();
    auto w <- p!.observe(x);
    p <- nil;
    return w;
  }

  /**
   * Evaluate the log probability density (or mass) function, if it exists.
   *
   * - x: The value.
   *
   * Return: the log probability density (or mass).
   */
  function logpdf(x:Value) -> Real {
    p <- p!.graft();
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
    p <- p!.graft();
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
    p <- p!.graft();
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
    p <- p!.graft();
    return p!.quantile(P);
  }
  
  /**
   * Finite lower bound of the support of this node, if any.
   */
  function lower() -> Value? {
    p <- p!.graft();
    return p!.lower();
  }
  
  /**
   * Finite upper bound of the support of this node, if any.
   */
  function upper() -> Value? {
    p <- p!.graft();
    return p!.upper();
  }

  function graftGaussian() -> Gaussian? {
    if !hasValue() {
      auto q <- p!.graftGaussian();
      if q? {
        p <- Distribution<Value>?(q);
      }
    }
    return Gaussian?(p);
  }
    
  function graftBeta() -> Beta? {
    if !hasValue() {
      auto q <- p!.graftBeta();
      if q? {
        p <- Distribution<Value>?(q);
      }
    }
    return Beta?(p);
  }
  
  function graftGamma() -> Gamma? {
    if !hasValue() {
      auto q <- p!.graftGamma();
      if q? {
        p <- Distribution<Value>?(q);
      }
    }
    return Gamma?(p);
  }
  
  function graftInverseGamma() -> InverseGamma? {
    if !hasValue() {
      auto q <- p!.graftInverseGamma();
      if q? {
        p <- Distribution<Value>?(q);
      }
    }
    return InverseGamma?(p);
  } 

  function graftIndependentInverseGamma() -> IndependentInverseGamma? {
    if !hasValue() {
      auto q <- p!.graftIndependentInverseGamma();
      if q? {
        p <- Distribution<Value>?(q);
      }
    }
    return IndependentInverseGamma?(p);
  } 

  function graftInverseWishart() -> InverseWishart? {
    if !hasValue() {
      auto q <- p!.graftInverseWishart();
      if q? {
        p <- Distribution<Value>?(q);
      }
    }
    return InverseWishart?(p);
  } 
  
  function graftNormalInverseGamma() -> NormalInverseGamma? {
    if !hasValue() {
      auto q <- p!.graftNormalInverseGamma();
      if q? {
        p <- Distribution<Value>?(q);
      }
    }
    return NormalInverseGamma?(p);
  }
  
  function graftDirichlet() -> Dirichlet? {
    if !hasValue() {
      auto q <- p!.graftDirichlet();
      if q? {
        p <- Distribution<Value>?(q);
      }
    }
    return Dirichlet?(p);
  }

  function graftRestaurant() -> Restaurant? {
    if !hasValue() {
      auto q <- p!.graftRestaurant();
      if q? {
        p <- Distribution<Value>?(q);
      }
    }
    return Restaurant?(p);
  }

  function graftMultivariateGaussian() -> MultivariateGaussian? {
    if !hasValue() {
      auto q <- p!.graftMultivariateGaussian();
      if q? {
        p <- Distribution<Value>?(q);
      }
    }
    return MultivariateGaussian?(p);
  }

  function graftMultivariateNormalInverseGamma() ->
      MultivariateNormalInverseGamma? {
    if !hasValue() {
      auto q <- p!.graftMultivariateNormalInverseGamma();
      if q? {
        p <- Distribution<Value>?(q);
      }
    }
    return MultivariateNormalInverseGamma?(p);
  }

  function graftMatrixGaussian() -> MatrixGaussian? {
    if !hasValue() {
      auto q <- p!.graftMatrixGaussian();
      if q? {
        p <- Distribution<Value>?(q);
      }
    }
    return MatrixGaussian?(p);
  }

  function graftMatrixNormalInverseGamma() -> MatrixNormalInverseGamma? {
    if !hasValue() {
      auto q <- p!.graftMatrixNormalInverseGamma();
      if q? {
        p <- Distribution<Value>?(q);
      }
    }
    return MatrixNormalInverseGamma?(p);
  }

  function graftMatrixNormalInverseWishart() -> MatrixNormalInverseWishart? {
    if !hasValue() {
      auto q <- p!.graftMatrixNormalInverseWishart();
      if q? {
        p <- Distribution<Value>?(q);
      }
    }
    return MatrixNormalInverseWishart?(p);
  }

  function graftDiscrete() -> Discrete? {
    if !hasValue() {
      auto q <- p!.graftDiscrete();
      if q? {
        p <- Distribution<Value>?(q);
      }
    }
    return Discrete?(p);
  }

  function graftBoundedDiscrete() -> BoundedDiscrete? {
    if !hasValue() {
      auto q <- p!.graftBoundedDiscrete();
      if q? {
        p <- Distribution<Value>?(q);
      }
    }
    return BoundedDiscrete?(p);
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
