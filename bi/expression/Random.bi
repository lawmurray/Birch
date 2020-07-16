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
final class Random<Value> < Expression<Value>(nil) {  
  /**
   * Associated distribution.
   */
  p:Distribution<Value>?;
  
  /**
   * Accumulated gradient.
   */
  d:Value?;

  /**
   * Value assignment.
   */
  operator <- x:Value {
    assert !this.x?;
    assert !this.p?;
    this.x <- x;
    constant();
  }

  /**
   * Optional assignment.
   */
  operator <- x:Value? {
    if x? {
      this <- x!;
    }
  }

  override function distribution() -> Distribution<Value>? {
    return p;
  }

  override function isRandom() -> Boolean {
    return true;
  }

  /**
   * Does this have a distribution?
   */
  function hasDistribution() -> Boolean {
    return p?;
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
    p.setRandom(this);
    this.p <- p;
  }

  override function doDepth() -> Integer {
    return 1;
  }
  
  override function doRows() -> Integer {
    return p!.rows();
  }

  override function doColumns() -> Integer {
    return p!.columns();
  }

  override function doValue() -> Value {
    assert p?;
    p!.prune();
    auto x <- p!.simulate();
    p!.update(x);
    p!.unlink();
    p!.unsetRandom(this);
    return x;
  }

  override function doPilot(gen:Integer) -> Value {
    return doGet();
  }

  override function doGet() -> Value {
    assert p?;
    if p!.supportsLazy() {
      p!.prune();
      auto x <- p!.simulateLazy();
      assert x?;
      p!.updateLazy(this);
      p!.unlink();
      p!.unsetRandom(this);
      return x!;
    } else {
      return doValue();
    }
  }

  override function doMove(gen:Integer, κ:Kernel) -> Value {
    return κ.move(this);
  }

  override function doPrior() -> Expression<Real>? {
    if p? {
      auto p1 <- p!.logpdfLazy(this);
      p <- nil;
      if p1? {
        auto p2 <- p1!.prior();
        if p2? {
          return p1! + p2!;
        } else {
          return p1!;
        }
      }
    }
    return nil;
  }

  override function doCompare(gen:Integer, x:DelayExpression, κ:Kernel) ->
      Real {
    auto y <- Random<Value>?(x)!;
    return κ.logpdf(y, this) - κ.logpdf(this, y);
  }

  function doAccumulateGrad(d:Value) {
    // ^ override is committed from the declaration as Value is not
    //   necessarily one of the supported gradient types in Expression (and
    //   need not be for this to work correctly)
    if this.d? {
      this.d <- this.d! + d;
    } else {
      this.d <- d;
    }
  }

  override function doGrad(gen:Integer) {
    //  
  }

  override function doClearGrad() {
    d <- nil;
  }

  override function doCount(gen:Integer) {
    //
  }
  
  override function doConstant() {
    //
  }

  override function doDetach() {
    p <- nil;
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

  override function graftMatrixNormalInverseWishart(compare:Distribution<LLT>) ->
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
    this <- buffer.get(x);
  }

  function write(buffer:Buffer) {
    if hasValue() || hasDistribution() {
      buffer.set(value());
    } else {
      buffer.setNil();
    }
  }
}
