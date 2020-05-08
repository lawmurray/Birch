/*
 * Expression state type.
 */
type ExpressionState = Integer8;

/*
 * Initial state for an Expression.
 */
EXPRESSION_INITIAL:ExpressionState <- 0;

/*
 * Pilot state for an Expression.
 */
EXPRESSION_PILOT:ExpressionState <- 1;

/*
 * Gradient state for an Expression.
 */
EXPRESSION_GRADIENT:ExpressionState <- 2;

/*
 * Value state for an Expression.
 */
EXPRESSION_VALUE:ExpressionState <- 3;

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
 * this purpose expressions are *stateful*, with the following state machine:
 *
 * <center>
 * <object type="image/svg+xml" data="../../figs/ExpressionState.svg"></object>
 * </center>
 *
 * [Random](../../classes/Random/) objects that occur in the expression are
 * treated as *variables* or *constants* according to these states. A Random
 * object in the pilot state is considered a variables, meaning that calls to
 * `grad()` will compute gradients with respect to it, and a further call to
 * `move()` will apply a Markov kernel to update its value. Random objects in
 * the value state are considered *constants*.
 */
abstract class Expression<Value> {  
  /**
   * Memoized value.
   */
  x:Value?;
  
  /**
   * Accumulated upstream gradient.
   */
  dfdx:Value?;
  
  /**
   * Count of the number of times `pilot()` has been called on this, minus
   * the number of times `grad()` has been called. This is used to accumulate
   * upstream gradients before recursing into a subexpression that may be
   * shared.
   */
  count:Integer <- 0;
  
  /**
   * Expression state.
   */
  state:ExpressionState <- EXPRESSION_INITIAL;

  /**
   * Value assignment. Once an expression has been assigned a value, it is
   * treated as though of type Boxed.
   */
  operator <- x:Value {
    this.x <- x;
  }

  /**
   * Length of result. This is equal to `rows()`.
   */
  final function length() -> Integer {
    return rows();
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
   * Get the memoized value of the expression, assuming it has already been
   * computed by a call to `value()` or `pilot()`.
   */
  final function get() -> Value {
    return x!;
  }

  /**
   * Evaluate and sever. Evaluates the expression and severs any references
   * to subexpressions, such as operands and arguments, as they are no longer
   * required.
   *
   * Returns: The evaluated value of the expression.
   *
   * Post-condition: The expression is in the value state.
   */
  final function value() -> Value {
    if !x? {
      doValue();
      state <- EXPRESSION_VALUE;
      assert x?;
    }
    assert state == EXPRESSION_VALUE;
    return x!;
  }
  
  /*
   * Evaluate and sever; overridden by derived classes.
   */
  abstract function doValue();

  /**
   * Evaluate and preserve. Evaluates the expression and preserves any
   * references to subexpressions, such as operands and arguments, as they
   * may yet be needed for gradient or Markov kernel computations.
   *
   * Pre-condition: The expression is in the initial or pilot state.
   *
   * Returns: The evaluated value of the expression.
   *
   * Post-condition: The expression is in the pilot state.
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
   *     Unless you are working on something like a gradient-based Markov
   *     kernel, you probably want to use `value()`, not `pilot()`. Doing
   *     otherwise may risk correctness. Consider the following:
   *
   *         if x.value() > 0.0 {
   *           doThis();
   *         } else {
   *           doThat();
   *         }
   *
   *     This is correct usage. Using `pilot()` instead of `value()` here
   *     may result in a subsequent move on the value of `x` (by e.g.
   *     `MoveParticleFilter`), without an adjustment on the branch taken.
   *     This would produce incorrect results.
   */
  final function pilot() -> Value {
    assert state == EXPRESSION_INITIAL || state == EXPRESSION_PILOT;
    count <- count + 1;
    if !x? {
      doPilot();
      state <- EXPRESSION_PILOT;
      assert x?;
    }
    assert state == EXPRESSION_PILOT;
    return x!;
  }
  
  /*
   * Evaluate and preserve; overridden by derived classes.
   */
  abstract function doPilot();

  /**
   * Evaluate gradients. Gradients are computed with respect to all
   * variables (Random objects in the pilot or gradient state).
   *
   * - d: Upstream gradient. For an initial call, this should be the unit for
   *     the given type, e.g. 1.0, a vector of ones, or the identity matrix.
   *
   * Pre-condition: The expression is in the pilot state.
   *
   * Post-condition: The expression is in the pilot or gradient state.
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
  function grad(d:Value) {
    assert state == EXPRESSION_PILOT;
    assert count >= 1;
    
    /* accumulate gradient */
    if dfdx? {
      dfdx <- add(dfdx!, d);
    } else {
      dfdx <- d;
    }
    
    /* continue recursion if all upstream gradients accumulated */
    count <- count - 1;
    if count == 0 {
      doGrad(dfdx!);
      state <- EXPRESSION_GRADIENT;
    }
    assert state == EXPRESSION_PILOT || state == EXPRESSION_GRADIENT;
  }
  
  /*
   * Evaluate gradient; overridden by derived classes;
   */
  abstract function doGrad(d:Value);

  /**
   * Move and re-evaluate. Variables are updated with a Markov kernel,
   * possibly using gradient information, and the expression re-evaluated
   * with these new values.
   *
   * Pre-condition: The expression is in the pilot or gradient state.
   *
   * Returns: The evaluated value of the expression.
   *
   * Post-condition: The expression is in the pilot state.
   */
  function move() -> Value {
    assert state == EXPRESSION_PILOT || state == EXPRESSION_GRADIENT;
    count <- count + 1;
    if state == EXPRESSION_GRADIENT {
      doMove();
      state <- EXPRESSION_PILOT;
    }
    assert state == EXPRESSION_PILOT;
    return x!;
  }

  /*
   * Move and re-evaluate; overridden by derived classes.
   */
  abstract function doMove();

  /**
   * Construct a lazy expression for the log-prior. This examines the
   * entire expression for moveable variables (Random objects) that have not
   * already been included in a log-prior expression (as indicated by a flag
   * in each Random object), and constructs a lazy expression that evaluates
   * to the log-prior.
   *
   * Pre-condition: the expression is in the pilot or gradient state.
   *
   * Returns: the log-prior expression, or nil if no such Random objectss
   * exist, which may be interpreted as the log-prior evaluating to zero.
   */
  function prior() -> Expression<Real>? {
    assert state == EXPRESSION_PILOT || state == EXPRESSSION_GRADIENT;
    return doPrior();
  }

  /*
   * Evaluate prior; overridden by derived classes;
   */
  abstract function doPrior() -> Expression<Real>?;

  /**
   * Set the value.
   *
   * - x: The value.
   *
   * Post-condition: The expression is in the value state.
   */
  final function setValue(x:Value) {
    assert !this.x?;
    this.x <- x;
    doSetValue();
    state <- EXPRESSION_VALUE;
  }
  
  /*
   * Propose a value; overridden by derived classes if supported (notably
   * Random).
   */
  function doSetValue() {
    assert false;
  }

  /**
   * Propose a pilot value.
   *
   * - x: The value.
   *
   * Pre-condition: The expression is in the initial state.
   *
   * Post-condition: The expression is in the pilot state.
   */
  final function setPilot(x:Value) {
    assert state == EXPRESSION_INITIAL;
    assert !this.x?;
    this.x <- x;
    doSetPilot();
    state <- EXPRESSION_PILOT;
  }
  
  /*
   * Propose a pilot value; overridden by derived classes if supported
   * (notably Random).
   */
  function doSetPilot() {
    assert false;
  }

  /**
   * If this is a Random, get the distribution associated with it, if any,
   * otherwise nil.
   */
  function distribution() -> Distribution<Value>? {
    return nil;
  }

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
  function graftNormalInverseGamma(compare:Distribution<Real>) -> NormalInverseGamma? {
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

/**
 * Length of a vector.
 */
function length(x:Expression<Real[_]>) -> Integer {
  return x.length();
}

/**
 * Number of rows of a vector; equals `length()`.
 */
function rows(x:Expression<Real[_]>) -> Integer {
  return x.rows();
}

/**
 * Number of columns of a vector; equals 1.
 */
function columns(x:Expression<Real[_]>) -> Integer {
  return x.columns();
}

/**
 * Length of a matrix; equals `rows()`.
 */
function length(x:Expression<Real[_,_]>) -> Integer {
  return x.length();
}

/**
 * Number of rows of a matrix.
 */
function rows(x:Expression<Real[_,_]>) -> Integer {
  return x.rows();
}

/**
 * Number of columns of a matrix.
 */
function columns(x:Expression<Real[_,_]>) -> Integer {
  return x.columns();
}

/**
 * Length of a vector.
 */
function length(x:Expression<Integer[_]>) -> Integer {
  return x.length();
}

/**
 * Number of rows of a vector; equals `length()`.
 */
function rows(x:Expression<Integer[_]>) -> Integer {
  return x.rows();
}

/**
 * Number of columns of a vector; equals 1.
 */
function columns(x:Expression<Integer[_]>) -> Integer {
  return x.columns();
}

/**
 * Length of a matrix; equals `rows()`.
 */
function length(x:Expression<Integer[_,_]>) -> Integer {
  return x.length();
}

/**
 * Number of rows of a matrix.
 */
function rows(x:Expression<Integer[_,_]>) -> Integer {
  return x.rows();
}

/**
 * Number of columns of a matrix.
 */
function columns(x:Expression<Integer[_,_]>) -> Integer {
  return x.columns();
}
