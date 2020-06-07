/**
 * Lazy expression.
 *
 * - Value: Value type.
 *
 * The simplest use case of lazy expressions is to simply construct them and
 * then call `value()` to evaluate them. Evaluations are memoized, so that
 * subsequent calls to `value()` do not require re-evaluation of the
 * expression, they simply return the memoized value. Objects may be reused
 * in multiple expressions, e.g. two expressions with a common subexpression
 * can share that subexpression.
 *
 * More elaborate use cases include computing gradients and applying Markov
 * kernels. These may require multiple calls to member functions, and for
 * this purpose expressions are stateful:
 *
 * - An Expression is initially considered *variable*.
 * - Once `value()` is called on the Expression, it and any subexpressions
 *   are considered *constant*.
 *
 * This particularly affects the `grad()` and `move()` operations:
 *
 * - `grad()` will compute gradients with respect to any
 *   [Random](../classes/Random/) objects considered variables, and similarly,
 * - `move()` will apply a Markov kernel to any [Random](../classes/Random/)
 *   objects considered variables.
 */
abstract class Expression<Value> < DelayExpression {  
  /**
   * Memoized value.
   */
  x:Value?;

  /**
   * Get the memoized value of the expression, assuming it has already been
   * computed by a call to `value()` or `pilot()`.
   */
  final function get() -> Value {
    return x!;
  }

  /**
   * If this is a Random, get the distribution associated with it, if any,
   * otherwise nil.
   */
  function distribution() -> Distribution<Value>? {
    return nil;
  }

  /**
   * Evaluate and make constant.
   *
   * Returns: The evaluated value of the expression.
   *
   * An expression is considered constant once `value()` has been called on
   * it. All subexpressions are necessarily also then constant, including any
   * Random objects that occur, which are no longer considered variables for
   * the purpose of `grad()` and `move()`.
   *
   * If this is not the intended behavior, consider `pilot()`.
   */
  final function value() -> Value {
    if !flagValue {
      if !x? {
        doValue();
      } else {
        doMakeConstant();
      }
      flagValue <- true;
    }
    return x!;
  }
  
  /*
   * Evaluate and make constant for `value()`.
   */
  abstract function doValue();

  /*
   * Make constant. This is used when a value has already been evaluated
   * (e.g. with `pilot()`), to make the expression constant.
   */
  final function makeConstant() {
    assert x?;
    if !flagValue {
      flagValue <- true;
      doMakeConstant();
    }
  }

  /*
   * Make constant for `value()`.
   */
  abstract function doMakeConstant();

  /**
   * Evaluate with count.
   *
   * Returns: The evaluated value of the expression.
   *
   * `pilot()` may be called multiple times, which accumulates a count. If
   * the intent is to compute gradients, `grad()` must subsequently be called
   * the same number of times. This is because subexpressions may be shared.
   * The calls to `pilot()` are used to determine how many times the object
   * occurs as a subexpression. The subsequent calls to `grad()` accumulate
   * upstream gradients this number of times before recursing. This is an
   * important optimization for many use cases, such as computing gradients
   * through Bayesian updates.
   *
   * !!! caution
   *     Unless you are working on something like a Markov kernel, you
   *     probably want to use `value()`, not `pilot()`. Doing otherwise may
   *     risk correctness. Consider the following:
   *
   *         if x.value() > 0.0 {
   *           doThis();
   *         } else {
   *           doThat();
   *         }
   *
   *     This is correct usage. Using `pilot()` instead of `value()` here
   *     may result in a subsequent move on the value of `x` (by e.g.
   *     `MoveParticleFilter`), without an adjustment on the branch taken,
   *     resulting in an invalid trace and incorrect results.
   */
  final function pilot() -> Value {
    if count == 0 {
      if !x? {
        doPilot();
      } else {
        doRestoreCount();
      }
    }
    count <- count + 1;
    return x!;
  }
  
  /*
   * Evaluate and count for `pilot()`.
   */
  abstract function doPilot();
  
  /*
   * Restore count. This is used when a value has already been evaluated
   * (e.g. with `pilot()`), but a subsequent call to e.g. `grad()` has
   * updated the count. This call restores it.
   */
  final function restoreCount() {
    assert x?;
    if count == 0 {
      doRestoreCount();
    }
    count <- count + 1;
  }
  
  /*
   * Count for `pilot()`.
   */
  abstract function doRestoreCount();

  /**
   * Evaluate gradients. Gradients are computed with respect to all
   * variables (Random objects in the pilot or gradient state).
   *
   * - Gradient: Gradient type. Must be one of `Real`, `Real[_]` or
   *   `Real[_,_]`.
   *
   * - d: Upstream gradient. For an initial call, this should be the unit for
   *      the given type, e.g. 1.0, a vector of ones, or the identity matrix.
   *
   * `grad()` must be called as many times as `pilot()` was previously
   * called. This is because subexpressions may be shared. The calls to
   * `pilot()` are used to determine how many times the object occurs as a
   * subexpression. The subsequent calls to `grad()` accumulate upstream
   * gradients this number of times before recursing. This is an important
   * optimization for many use cases, such as computing gradients through
   * Bayesian updates.
   *
   * Reverse-mode automatic differentiation is used. The previous calls to
   * `pilot()` constitute the forward pass, the subsequent calls to `grad()`
   * constitute the backward pass. If the expression tree encodes
   * $$x_n = f(x_0) = (f_n \circ \cdots \circ f_1)(x_0),$$
   * and this particular node encodes one of those functions
   * $x_i = f_i(x_{i-1})$, the argument to the function is
   * $$\frac{\partial (f_n \circ \cdots \circ f_{i+1})}
   * {\partial x_i}\left(x_i\right),$$
   * it computes
   * $$\frac{\partial (f_n \circ \cdots \circ f_{i})}
   * {\partial x_{i-1}}\left(x_{i-1}\right),$$
   * and passes the result to its child, which encodes $f_{i-1}$, to continue
   * the computation. The variable (Random object) that encodes $x_0$ keeps
   * the final result.
   */
  final function grad<Gradient>(d:Gradient) {
    if !flagValue {
      doAccumulateGrad(d);
      assert count > 0;
      count <- count - 1;
      if count == 0 {
        /* all upstream gradients accumulated, continue recursion */
        doGrad();
      }
    }
  }

  /*
   * Accumulate gradient.
   */
  function doAccumulateGrad(d:Real) {
    assert false;
  }
  
  /*
   * Accumulate gradient.
   */
  function doAccumulateGrad(d:Real[_]) {
    assert false;
  }
  
  /*
   * Accumulate gradient.
   */
  function doAccumulateGrad(D:Real[_,_]) {
    assert false;
  }

  /*
   * Clear accumulated gradient.
   */
  abstract function doClearGrad();
  
  /*
   * Evaluate gradient.
   */
  abstract function doGrad();

  /**
   * Move and re-evaluate. Variables are updated with a Markov kernel,
   * possibly using gradient information, and the expression re-evaluated
   * with these new values.
   *
   * - κ: Markov kernel.
   *
   * Returns: The evaluated value of the expression.
   */
  final function move(κ:Kernel) -> Value {
    if !flagValue {
      if count == 0 {
        doMove(κ);
        doClearGrad();
      }
      count <- count + 1;
    }
    return x!;
  }

  /*
   * Move and re-evaluate.
   */
  abstract function doMove(κ:Kernel);
  
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
  function graftNormalInverseGamma(compare:Distribution<Real>) ->
      NormalInverseGamma? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftLinearNormalInverseGamma(compare:Distribution<Real>) ->
      TransformLinear<NormalInverseGamma>? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftDotNormalInverseGamma(compare:Distribution<Real>) ->
      TransformDot<MultivariateNormalInverseGamma>? {
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
  function graftMultivariateNormalInverseGamma(compare:Distribution<Real>) ->
      MultivariateNormalInverseGamma? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftLinearMultivariateNormalInverseGamma(compare:Distribution<Real>) ->
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
   * - σ2: The inverse gamma distribution that must match to be successful.
   *
   * Return: The node if successful, nil if not.
   */
  function graftMatrixNormalInverseGamma(compare:Distribution<Real[_]>) ->
      MatrixNormalInverseGamma? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * - σ2: The inverse gamma distribution that must match to be successful.
   *
   * Return: The node if successful, nil if not.
   */
  function graftLinearMatrixNormalInverseGamma(compare:Distribution<Real[_]>) ->
      TransformLinearMatrix<MatrixNormalInverseGamma>? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftMatrixNormalInverseWishart(compare:Distribution<Real[_,_]>) ->
      MatrixNormalInverseWishart? {
    return nil;
  }

  /*
   * Attempt to graft this expression onto the delayed sampling graph.
   *
   * Return: The node if successful, nil if not.
   */
  function graftLinearMatrixNormalInverseWishart(compare:Distribution<Real[_,_]>) ->
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
