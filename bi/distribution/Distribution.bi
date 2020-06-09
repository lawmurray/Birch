/**
 * Distribution with delayed-sampling support.
 *
 * - Value: Value type.
 */
abstract class Distribution<Value> < DelayDistribution {
  /**
   * Random variate associated with the distibution, if it is on the delayed
   * sampling $M$-path.
   */
  x:Random<Value>&?;

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

  /**
   * Are lazy operations supported?
   */
  function supportsLazy() -> Boolean {
    return false;
  }

  /**
   * Returns `this`; a convenience for code generation within the compiler.
   */
  final function distribution() -> Distribution<Value> {
    return this;
  }

  /**
   * Set the random variate associated with the distribution, if any.
   */
  final function setRandom(x:Random<Value>) {
    assert !this.x?;
    this.x <- x;
  }

  /**
   * Realize the random variate associated with the distribution.
   */
  final function realize() {
    if supportsLazy() {
      x!.pilot();
    } else {
      x!.value();
    }
  }

  /**
   * Simulate a value for a random variate associated with this node,
   * updating the delayed sampling graph accordingly, and returning the
   * value.
   */
  function value() -> Value {
    prune();
    auto x <- simulate();
    update(x);
    unlink();
    return x;
  }
  
  /**
   * Observe a value for a random variate associated with this node,
   * updating the delayed sampling graph accordingly, and returning a weight
   * giving the log pdf (or pmf) of that variate under the distribution.
   */
  final function observe(x:Value) -> Real {
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
  final function observeLazy(x:Expression<Value>) -> Expression<Real>? {
    assert supportsLazy();
    prune();
    auto w <- logpdfLazy(x);
    updateLazy(x);
    unlink();
    return w;
  }
  
  /**
   * Simulate a value.
   *
   * Return: the value.
   */
  abstract function simulate() -> Value;

  /**
   * Simulate a value as part of a lazy expression.
   *
   * Return: the value, if supported.
   */
  function simulateLazy() -> Value? {
    return nil;
  }

  /**
   * Evaluate the log probability density (or mass) function.
   *
   * - x: The value.
   *
   * Return: the log probability density (or mass).
   */
  abstract function logpdf(x:Value) -> Real;

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
  function update(x:Value) {
    //
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
  function graftMatrixNormalInverseWishart(compare:Distribution<LLT>) ->
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
