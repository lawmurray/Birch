/*
 * Type-specific interface for delayed sampling $M$-path nodes.
 *
 * - Value: Value type.
 */
abstract class Distribution<Value> < Delay {
  /**
   * Value.
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
   * Does the node have a value?
   */
  function hasValue() -> Boolean {
    return x?;
  }

  /**
   * Realize a value for a random variate associated with this node.
   */
  function value() -> Value {
    if !x? {
      prune();
      x <- simulate();
      update(x!);
    }
    return x!;
  }

  /**
   * Set value.
   */
  function set(x:Value) {
    assert !this.x?;
    prune();
    this.x <- x;
    update(x);
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
    update(x);
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
      update(x.value());
    }
    return w;  
  }

  function realize() {
    value();
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
  function graftNormalInverseGamma() -> NormalInverseGamma? {
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
  function graftMultivariateNormalInverseGamma() -> MultivariateNormalInverseGamma? {
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
  function graftMatrixNormalInverseGamma() -> MatrixNormalInverseGamma? {
    return nil;
  }

  /**
   * Graft this onto the delayed sampling graph.
   */
  function graftMatrixNormalInverseWishart() -> MatrixNormalInverseWishart? {
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

  function write(buffer:Buffer) {
    buffer.set(value());
  }
}
