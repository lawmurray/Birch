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
   * Number of rows in result.
   */
  function rows() -> Integer {
    return 1;
  }
  
  /**
   * Number of columns in result.
   */
  function columns() -> Integer {
    return 1;
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
   * Set the child of any delayed sampling nodes in the expression.
   */
  abstract function setChild(child:Delay);

  /**
   * Graft this expression onto the delayed sampling graph.
   *
   * - child: The delayed sampling node that initiated the graft.
   */
  abstract function graft() -> Expression<Value>;

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftBeta() -> Beta? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftDirichlet() -> Dirichlet? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftRestaurant() -> Restaurant? {
    return nil;
  }
  
  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftGamma() -> Gamma? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftScaledGamma() ->  TransformLinear<Gamma>? {
    return nil;
  }
  
  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftInverseGamma() -> InverseGamma? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftIndependentInverseGamma() -> IndependentInverseGamma? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftInverseWishart() -> InverseWishart? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftGaussian() -> Gaussian? {
    return nil;
  }
  
  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftLinearGaussian() -> TransformLinear<Gaussian>? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftDotGaussian() -> TransformDot<MultivariateGaussian>? {
    return nil;
  }
  
  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftNormalInverseGamma() -> NormalInverseGamma? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftLinearNormalInverseGamma() ->
      TransformLinear<NormalInverseGamma>? {
    return nil;
  }


  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftMultivariateGaussian() -> MultivariateGaussian? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftLinearMultivariateGaussian() ->
      TransformLinearMultivariate<MultivariateGaussian>? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftMultivariateNormalInverseGamma() ->
      MultivariateNormalInverseGamma? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftLinearMultivariateNormalInverseGamma() ->
      TransformLinearMultivariate<MultivariateNormalInverseGamma>? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftMatrixGaussian() -> MatrixGaussian? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftLinearMatrixGaussian() ->
      TransformLinearMatrix<MatrixGaussian>? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftMatrixNormalInverseGamma() ->
      MatrixNormalInverseGamma? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftLinearMatrixNormalInverseGamma() ->
      TransformLinearMatrix<MatrixNormalInverseGamma>? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftMatrixNormalInverseWishart() ->
      MatrixNormalInverseWishart? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftLinearMatrixNormalInverseWishart() ->
      TransformLinearMatrix<MatrixNormalInverseWishart>? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftDiscrete() -> Discrete? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftBoundedDiscrete() -> BoundedDiscrete? {
    return nil;
  }
}
