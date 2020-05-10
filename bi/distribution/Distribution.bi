/**
 * Distribution with delayed-sampling support.
 *
 * - Value: Value type.
 */
abstract class Distribution<Value> < DelayDistribution {
  /**
   * Realized value, if the distribution's parent on the delayed sampling
   * $M$-path has forced its realization.
   */
  x:Value?;

  /**
   * Number of rows, when interpreted as a matrix.
   */
  function rows() -> Integer {
    return 1;
  }

  /**
   * Number of columns, when interpreted as a matrix.
   */
  function columns() -> Integer {
    return 1;
  }
  
  function value() -> Value {
    if isRealized() {
      return x!;
    } else {
      auto x <- simulate();
      update(x);
      unlink();
      return x;
    }
  }
  
  /**
   * Returns `this`; a convenience for code generation within the compiler.
   */
  function distribution() -> Distribution<Value> {
    return this;
  }
  
  /**
   * Observe a value for a random variate associated with this node,
   * updating the delayed sampling graph accordingly, and returning a weight
   * giving the log pdf (or pmf) of that variate under the distribution.
   */
  function observe(x:Value) -> Real {
    assert !this.x?;
    prune();
    auto w <- logpdf(x);
    if w > -inf {
      update(x);
    }
    unlink();
    return w;
  }
  
  /**
   * Observe a value for a random variate associated with this node,
   * updating the delayed sampling graph accordingly, and returning a lazy
   * expression giving the log pdf (or pmf) of that variate under the
   * distribution.
   */
  function observeLazy(x:Expression<Value>) -> Expression<Real>? {
    assert !this.x?;
    prune();
    auto w <- logpdfLazy(x);
    if w? {
      updateLazy(x);
    } else {
      w <- Boxed(logpdf(x.value()));
      update(x.value());
    }
    unlink();
    return w;  
  }

  /**
   * Realize a value from the distribution. This is used by the parent of the
   * distribution on the delayed sampling $M$-path if it must prune.
   */
  function realize() {
    if !x? {
      prune();
      x <- simulate();
      update(x!);
      unlink();
    }
  }
  
  /**
   * Is a value realized?
   */
  function isRealized() -> Boolean {
    return x?;
  }
  
  /**
   * Get the realized value.
   */
  function realized() -> Value {
    return x!;
  }
  
  /**
   * Simulate a value.
   *
   * Return: the value.
   */
  abstract function simulate() -> Value;

  /**
   * Evaluate the log probability density (or mass) function.
   *
   * - x: The value.
   *
   * Return: the log probability density (or mass).
   */
  abstract function logpdf(x:Value) -> Real;

  /**
   * Update the parent node on the $M$-path given the value of this node.
   *
   * - x: The value.
   */
  function update(x:Value) {
    //
  }

  /**
   * Downdate the parent node on the $M$-path given the value of this node.
   *
   * - x: The value.
   */
  function downdate(x:Value) {
    //
  }
  
  /**
   * Evaluate the probability density (or mass) function.
   *
   * - x: The value.
   *
   * Return: the probability density (or mass).
   */
  function pdf(x:Value) -> Real {
    return exp(logpdf(x));
  }

  /**
   * Evaluate the cumulative distribution function at a value.
   *
   * - x: The value.
   *
   * Return: the cumulative probability, if supported.
   */
  function cdf(x:Value) -> Real? {
    return nil;
  }

  /**
   * Evaluate the quantile function at a cumulative probability.
   *
   * - P: The cumulative probability.
   *
   * Return: the quantile, if supported.
   */
  function quantile(P:Real) -> Value? {
    return nil;
  }
  
  /**
   * Finite lower bound of the support of this node, if any.
   */
  function lower() -> Value? {
    return nil;
  }
  
  /**
   * Finite upper bound of the support of this node, if any.
   */
  function upper() -> Value? {
    return nil;
  }

  /**
   * Simulate a value as part of a lazy expression.
   *
   * Return: the value, if supported.
   */
  function simulateLazy() -> Value? {
    return nil;
  }

  /**
   * Construct a lazy expression for the log probability density (or mass).
   *
   * - x: The value.
   *
   * Return: the log probability density (or mass), if supported.
   */
  function logpdfLazy(x:Expression<Value>) -> Expression<Real>? {
    return nil;
  }

  /**
   * Update the parent node on the $M$-path given the value of this node.
   *
   * - x: The value.
   */
  function updateLazy(x:Expression<Value>) {
    //
  }

  /**
   * Graft this onto the delayed sampling graph.
   *
   * Returns: The object to attach to the delayed sampling graph. This may
   * be this object, or a substitute based on variable elimination rules.
   * Call `attach()` on the object to finalize the graft.
   */
  function graft() -> Distribution<Value> {
    prune();
    return this;
  }

  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftGaussian() -> Gaussian? {
    return nil;
  }
    
  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftBeta() -> Beta? {
    return nil;
  }
  
  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftGamma() -> Gamma? {
    return nil;
  }
  
  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftInverseGamma() -> InverseGamma? {
    return nil;
  } 

  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftIndependentInverseGamma() -> IndependentInverseGamma? {
    return nil;
  } 

  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftInverseWishart() -> InverseWishart? {
    return nil;
  } 
  
  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftNormalInverseGamma(compare:Distribution<Real>) -> NormalInverseGamma? {
    return nil;
  }
  
  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftDirichlet() -> Dirichlet? {
    return nil;
  }

  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftRestaurant() -> Restaurant? {
    return nil;
  }

  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftMultivariateGaussian() -> MultivariateGaussian? {
    return nil;
  }

  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftMultivariateNormalInverseGamma(compare:Distribution<Real>) ->
      MultivariateNormalInverseGamma? {
    return nil;
  }

  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftMatrixGaussian() -> MatrixGaussian? {
    return nil;
  }

  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftMatrixNormalInverseGamma(compare:Distribution<Real[_]>) ->
      MatrixNormalInverseGamma? {
    return nil;
  }

  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftMatrixNormalInverseWishart(compare:Distribution<Real[_,_]>) ->
      MatrixNormalInverseWishart? {
    return nil;
  }

  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftDiscrete() -> Discrete? {
    return nil;
  }

  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftBoundedDiscrete() -> BoundedDiscrete? {
    return nil;
  }
}
