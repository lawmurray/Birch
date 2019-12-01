/**
 * Abstract lazy expression.
 *
 * - Value: Value type.
 */
abstract class Expression<Value> {  
  /**
   * Value conversion.
   */
  operator -> Value {
    return value();
  }

  /**
   * Realized value.
   */
  abstract function value() -> Value;
  
  /**
   * Pilot value.
   */
  abstract function pilot() -> Value;

  /**
   * Proposed value. This is called after `pilot()` and perhaps `grad()`
   * to propose an alternative to the pilot position.
   */
  abstract function propose() -> Value;
  
  /**
   * Compute gradient. This is called after `pilot()` to evaluate the
   * gradient of the function with respect to Random objects at the pilot
   * position.
   *
   * This uses reverse-mode automatic differentiation. If the  expression
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
  abstract function dpilot(d:Value);

  /**
   * Compute gradient. This is called after `proposal()` to evaluate the
   * gradient of the function with respect to Random objects at the propoosal
   * position.
   *
   * See: `dpilot()`.
   */
  abstract function dpropose(d:Value);
  
  /**
   * Compute a partial acceptance ratio.
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
   * Accept the proposal. This is called after `propose()`, and sets the
   * value of all Random objects in the expression to their proposal
   * positions.
   */
  abstract function accept();

  /**
   * Reject the proposal. This is called after `propose()`, and sets the
   * value of all Random objects in the expression to their pilot positions.
   */
  abstract function reject();
  
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
