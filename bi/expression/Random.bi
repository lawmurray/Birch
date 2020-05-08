/**
 * Random variate.
 *
 * - Value: Value type.
 *
 * Random objects, like all [Expression](../../classes/Expression/) objects,
 * are stateful. Random objects in the pilot state are considered
 * *variables*, meaning that a call to `grad()` will compute gradients with
 * respect to them, and a further call to `move()` will apply a Markov kernel
 * to update their value. Random objects in the value state are considered
 * *constants*.
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
  
  override function rows() -> Integer {
    if x? {
      return global.rows(x!);
    } else {
      assert p?;
      return p!.rows();
    }
  }

  override function columns() -> Integer {
    if x? {
      return global.columns(x!);
    } else {
      assert p?;
      return p!.columns();
    }
  }

  override function distribution() -> Distribution<Value>? {
    return p;
  }

  /**
   * Does this have a value?
   */
  function hasValue() -> Boolean {
    return x? || (p? && p!.isRealized());
  }

  /**
   * Does this have a distribution?
   */
  function hasDistribution() -> Boolean {
    return p?;
  }

  override function doValue() {
    if p!.isRealized() {
      /* distribution was forced to realize by its parent, use the value that
       * was simulated */
      x <- p!.realized();
      p <- nil;
    } else {
      p <- p!.graft();
      x <- p!.simulate();
      p!.update(x!);
      p!.unlink();
      p <- nil;
    }
  }

  override function doPilot() {
    if p!.isRealized() {
      /* value must be considered escaped in this case, act the same as
       * doValue() */
      x <- p!.realized();
      p <- nil;
    } else {
      p <- p!.graft();
      x <- p!.simulateLazy();
      p!.updateLazy(this);
      p!.unlink();
      // keep p here, this distinguishes between whether or not a subsequent
      // gradient computation is possible
    }
  }

  override function doSetValue() {
    p!.prune();
    p!.update(x!);
    p!.unlink();
    p <- nil;
  }

  override function doSetPilot() {
    p!.prune();
    p!.updateLazy(this);
    p!.unlink();
  }

  override function doGrad(d:Value) {
    assert x?;
    if p? {
      auto ψ <- p!.logpdfLazy(this);
      if ψ? {
        w <- ψ!.pilot();
        ψ!.grad(1.0);
      } else {
        /* treat as a constant */
        dfdx <- nil;
      }
    } else {
      /* treat as a constant */
      dfdx <- nil;
    }
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

  override function graftGaussian() -> Gaussian? {
    if !hasValue() {
      auto q <- p!.graftGaussian();
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  }
    
  override function graftBeta() -> Beta? {
    if !hasValue() {
      auto q <- p!.graftBeta();
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  }
  
  override function graftGamma() -> Gamma? {
    if !hasValue() {
      auto q <- p!.graftGamma();
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  }
  
  override function graftInverseGamma() -> InverseGamma? {
    if !hasValue() {
      auto q <- p!.graftInverseGamma();
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  } 

  override function graftIndependentInverseGamma() -> IndependentInverseGamma? {
    if !hasValue() {
      auto q <- p!.graftIndependentInverseGamma();
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  } 

  override function graftInverseWishart() -> InverseWishart? {
    if !hasValue() {
      auto q <- p!.graftInverseWishart();
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  } 
  
  override function graftNormalInverseGamma(compare:Distribution<Real>) -> NormalInverseGamma? {
    if !hasValue() {
      auto q <- p!.graftNormalInverseGamma(compare);
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  }
  
  override function graftDirichlet() -> Dirichlet? {
    if !hasValue() {
      auto q <- p!.graftDirichlet();
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  }

  override function graftRestaurant() -> Restaurant? {
    if !hasValue() {
      auto q <- p!.graftRestaurant();
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  }

  override function graftMultivariateGaussian() -> MultivariateGaussian? {
    if !hasValue() {
      auto q <- p!.graftMultivariateGaussian();
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  }

  override function graftMultivariateNormalInverseGamma(compare:Distribution<Real>) ->
      MultivariateNormalInverseGamma? {
    if !hasValue() {
      auto q <- p!.graftMultivariateNormalInverseGamma(compare);
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  }

  override function graftMatrixGaussian() -> MatrixGaussian? {
    if !hasValue() {
      auto q <- p!.graftMatrixGaussian();
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  }

  override function graftMatrixNormalInverseGamma(compare:Distribution<Real[_]>) ->
      MatrixNormalInverseGamma? {
    if !hasValue() {
      auto q <- p!.graftMatrixNormalInverseGamma(compare);
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  }

  override function graftMatrixNormalInverseWishart(compare:Distribution<Real[_,_]>) ->
      MatrixNormalInverseWishart? {
    if !hasValue() {
      auto q <- p!.graftMatrixNormalInverseWishart(compare);
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  }

  override function graftDiscrete() -> Discrete? {
    if !hasValue() {
      auto q <- p!.graftDiscrete();
      p <-? Distribution<Value>?(q);
      return q;
    }
    return nil;
  }

  override function graftBoundedDiscrete() -> BoundedDiscrete? {
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
