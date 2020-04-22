/**
 * Distribution with delayed-sampling support.
 *
 * - Value: Value type.
 */
abstract class Distribution<Value> < Delay {
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
  
  /**
   * Observe a value for a random variate associated with this node,
   * updating (or downdating) the delayed sampling graph accordingly, and
   * returning a weight giving the log pdf (or pmf) of that variate under the
   * distribution.
   */
  function observe(x:Value) -> Real {
    assert !this.x?;
    prune();
    this.x <- x;
    auto w <- logpdf(x);
    if w > -inf {
      update(x);
    }
    unlink();
    return w;
  }
  
  /**
   * Observe a value for a random variate associated with this node,
   * updating (or downdating) the delayed sampling graph accordingly, and
   * returning a weight giving the log pdf (or pmf) of that variate under the
   * distribution. Additionally computes gradients for subsequent moves.
   */
  function observeLazy(x:Expression<Value>) -> Real {
    assert !this.x?;
    prune();
    this.x <- x.value();
    auto ψ <- logpdfLazy(x);
    auto w <- 0.0;
    if ψ? {
      w <- ψ!.value();
      ψ!.grad(1.0);
      updateLazy(x);
    } else {
      w <- logpdf(x.value());
      if w > -inf {
        update(x.value());
      }
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
    assert false;
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
    graftFinalize();
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
  
  /**
   * Finalize a graft onto the delayed sampling $M$-path. Use this on an
   * object returned by `graft*()` member functions as a final check.
   *
   * - Returns: True if the graft was successfully finalized, false
   *   otherwise.
   *
   * False is returned in a situation where an object is proposed due to a
   * matching template for variable elimination, but further checks determine
   * that the object is invalid. For example:
   *
   *     x ~ Gaussian(μ, σ2);
   *     y ~ Gaussian(x, x*x);
   *
   * This initially matches for a Gaussian-Gaussian conjugacy, but this
   * further check evaluates the variance of `y`, determining that this then
   * realizes `x`, and thus the conjugacy is no longer valid.
   */
  function graftFinalize() -> Boolean {
    return true;
  }
}
