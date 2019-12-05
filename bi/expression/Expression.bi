/**
 * Abstract lazy expression.
 *
 * - Value: Value type.
 *
 * An expression can maintain up to three values at any one time:
 *
 * - a *final* value, 
 * - a *pilot* value,
 * - a *proposal* value.
 *
 * The final value is set by assignment, implicit conversion to the Value
 * type, explicit conversion to the Value type using `value()`, or final
 * acceptance of a proposed value using `clamp()`. Once set, the other values
 * are irrelevant, and the expression behaves as though a Boxed value.
 *
 * The pilot value is set using `pilot()` function. It is temporary, and used
 * when a Markov kernel may be applied to later update the value. The
 * proposal value is used during the computation of that Markov kernel. After
 * one or more applications of Markov kernels, the pilot value can be locked
 * in as the final value with `clamp()`.
 */
abstract class Expression<Value> {  
  /**
   * Value conversion.
   */
  operator -> Value {
    return value();
  }

  /**
   * Value assignment. Once an expression has been assigned a value, it is
   * treated as though of type Boxed.
   */
  operator <- x:Value {
    assert false;
  }

  /**
   * Value computation.
   */
  abstract function value() -> Value;

  /**
   * Pilot value computation.
   */
  abstract function pilot() -> Value;

  /**
   * Proposal value computation.
   */
  abstract function propose() -> Value;

  /**
   * Compute gradients of the expression with respect to all Random objects,
   * at the pilot value.
   *
   * - d: Upstream gradient. For an initial call, this should be the unit for
   *     the given type, e.g. 1.0, 1, true, a vector of ones, or the identity
   *     matrix.
   *
   * Returns: Are there one or more Random objects with non-zero gradients?
   *
   * This uses reverse-mode automatic differentiation. If the expression
   * tree encodes
   * $$x_n = f(x_0) = (f_n \circ \cdots \circ f_1)(x_0),$$
   * and this particular node encodes one of those functions
   * $x_i = f_i(x_{i-1})$, the argument to the function is
   * $$\frac{\partial (f_n \circ \cdots \circ f_{i+1})}
   * {\partial x_i}\left(x_i\right),$$
   * it computes
   * $$\frac{\partial (f_n \circ \cdots \circ f_{i})}
   * {\partial x_{i-1}}\left(x_{i-1}\right),$$
   * and passes the result to its child, which encodes $f_{i-1}$, to continue
   * the computation. The Random object that encodes $x_0$ keeps the final
   * result.
   */
  abstract function gradPilot(d:Value) -> Boolean;

  /**
   * Compute gradients of the expression with respect to all Random objects,
   * at the proposal value.
   *
   * See also: `gradPilot()`.
   */
  abstract function gradPropose(d:Value) -> Boolean;

  /**
   * Sum contributions to the logarithm of the acceptance ratio.
   *
   * Returns: The quantity:
   * $$\log \left(\frac{p(x^\prime) q(x^\star \mid x^\prime)}
   * {p(x^\star) q(x^\prime \mid x^\star)}\right),$$
   * where $x^\star$ represents the pilot position and $x^\prime$ the proposal
   * position of all Random objects in the expression, $p$ the prior
   * distribution and $q$ the proposal distribution of the same Random
   * objects. This quantity forms part of the Metropolis--Hastings acceptance
   * ratio to determine whether to accept or reject the proposal position in
   * favour of the pilot.
   */
  abstract function ratio() -> Real;

  /**
   * Accept the proposal value. The pilot value is set to the proposal
   * value, and the proposal value discarded.
   */
  abstract function accept();
  
  /**
   * Reject the proposal value. The pilot value is preserved and the proposal
   * value discarded.
   */
  abstract function reject();
  
  /**
   * Set the final value to the pilot value. The pilot value is discarded.
   */
  abstract function clamp();

  /**
   * If this expression is grafted onto the delayed sampling graph, get the
   * node with which it is associated on that graph.
   */
  function getDelay() -> Delay? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftBeta() -> DelayBeta? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftDirichlet() -> DelayDirichlet? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftRestaurant() -> DelayRestaurant? {
    return nil;
  }
  
  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftGamma() -> DelayGamma? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftScaledGamma() ->  TransformLinear<DelayGamma>? {
    return nil;
  }
  
  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftInverseGamma() -> DelayInverseGamma? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftIndependentInverseGamma() -> DelayIndependentInverseGamma? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftInverseWishart() -> DelayInverseWishart? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftGaussian() -> DelayGaussian? {
    return nil;
  }
  
  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftLinearGaussian() -> TransformLinear<DelayGaussian>? {
    return nil;
  }
  
  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftNormalInverseGamma() -> DelayNormalInverseGamma? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftLinearNormalInverseGamma() ->
      TransformLinear<DelayNormalInverseGamma>? {
    return nil;
  }


  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftMultivariateGaussian() -> DelayMultivariateGaussian? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftLinearMultivariateGaussian() ->
      TransformLinearMultivariate<DelayMultivariateGaussian>? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftMultivariateNormalInverseGamma() ->
      DelayMultivariateNormalInverseGamma? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftLinearMultivariateNormalInverseGamma() ->
      TransformLinearMultivariate<DelayMultivariateNormalInverseGamma>? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftMatrixGaussian() -> DelayMatrixGaussian? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftLinearMatrixGaussian() ->
      TransformLinearMatrix<DelayMatrixGaussian>? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftMatrixNormalInverseGamma() ->
      DelayMatrixNormalInverseGamma? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftLinearMatrixNormalInverseGamma() ->
      TransformLinearMatrix<DelayMatrixNormalInverseGamma>? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftMatrixNormalInverseWishart() ->
      DelayMatrixNormalInverseWishart? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftLinearMatrixNormalInverseWishart() ->
      TransformLinearMatrix<DelayMatrixNormalInverseWishart>? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftDiscrete() -> DelayDiscrete? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftBoundedDiscrete() -> DelayBoundedDiscrete? {
    return nil;
  }
}
