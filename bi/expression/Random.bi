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
   * Piloted value.
   */
  x':Value?;
  
  /**
   * Gradient at the piloted value.
   */
  dfdx':Value?;
  
  /**
   * Proposed value.
   */
  x'':Value?;
  
  /**
   * Gradient at the proposed value.
   */
  dfdx'':Value?;
  
  /**
   * Contribution to the log acceptance ratio?
   */
  α:Real?;

  /**
   * Value assignment.
   */
  operator <- x:Value {
    assert !dist?;
    assert !x'?;
    assert !x''?;
    this.x <- x;
  }

  /**
   * Optional value assignment.
   */
  operator <- x:Value? {
    assert !dist?;
    assert !x'?;
    assert !x''?;
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
    assert !x'?;
    assert !x''?;
    return x!;
  }

  function pilot() -> Value {
    if x? {
      return x!;
    } else {
      assert dist?;
      if !x'? {
        x' <- dist!.pilot();
      }
      return x'!;
    }
  }

  function gradPilot(d:Value) -> Boolean {
    if x? {
      return false;
    } else {
      assert dist?;
      assert x'?;
      if !dfdx'? {
        /* first time this has been encountered in the gradient computation,
         * propagate into its prior */
        dfdx' <- d;
        auto p <- dist!.lazy(this);
        α <- -p!.pilot();
        p!.gradPilot(1.0);
      } else {
        /* second or subsequent time this has been encountered in the gradient
         * computation; accumulate */
        dfdx' <- dfdx'! + d;
      }
      return dfdx'?;
    }
  }

  function propose() -> Value {
    if x? {
      return x!;
    } else {
      assert dist?;
      assert x'?;
      if !x''? {
        x'' <- dist!.propose();  // simulate to recurse through prior but...
        if dfdx'? {
          x'' <- simulate_propose(x'!, dfdx'!);  // ...replace with local proposal
          α <- α! - logpdf_propose(x''!, x'!, dfdx'!);
        }
      }
      return x''!;
    }
  }
  
  function gradPropose(d:Value) -> Boolean {
    if x? {
      return false;
    } else {
      assert dist?;
      assert x''?;
      if dfdx'? && !dfdx''? {
        /* first time this has been encountered in the gradient computation,
         * propagate into its prior */
        dfdx'' <- d;
        auto p <- dist!.lazy(this);
        α <- α! + p!.propose();
        p!.gradPropose(1.0);
      } else {
        /* second or subsequent time this has been encountered in the gradient
         * computation; accumulate */
        dfdx'' <- dfdx''! + d;
      }
      return dfdx''?;
    }
  }
  
  function ratio() -> Real {
    if α? {
      if dfdx''? {
        α <- α! + logpdf_propose(x'!, x''!, dfdx''!);
      }
      auto result <- α!;
      α <- nil;
      return result;
    } else {
      return 0.0;
    }
  }
  
  function accept() {
    if x? {
      // nothing to do
    } else if x''? {
      x' <- x'';
      dfdx' <- dfdx'';
      x'' <- nil;
      dfdx'' <- nil;
      α <- nil;
      dist!.lazy(this)!.accept();
    }
  }

  function reject() {
    if x? {
      // nothing to do
    } else if x''? {
      x'' <- nil;
      dfdx'' <- nil;
      α <- nil;
      dist!.lazy(this)!.reject();
    }
  }

  function clamp() {
    if x? {
      // nothing to do
    } else {
      x <- x';
      x' <- nil;
      dfdx' <- nil;
      x'' <- nil;
      dfdx'' <- nil;
      α <- nil;
      dist!.set(x!);
      dist!.lazy(this)!.clamp();
      dist <- nil;
    }
  }

  function graft(child:Delay) -> Expression<Value> {
    if x? {
      return Boxed(x!);
    } else {
      return dist!.graft(child);
    }
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
