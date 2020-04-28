/**
 * Abstract lazy expression.
 *
 * - Value: Value type.
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
   * Evaluate.
   *
   * Returns: The evaluated value of the expression.
   */
  final function value() -> Value {
    if !x? {
      doValue();
      assert x?;
    }
    return x!;
  }
  
  /**
   * Evaluate; overridden by derived classes.
   */
  abstract function doValue();

  /**
   * Evaluate before `grad()`.
   *
   * Returns: The evaluated value of the expression.
   *
   * `pilot()` differs to `value()` in the following ways:
   *
   *   * For Random objects with no value but with a distribution, `pilot()`
   *     proposes a value by simulating from the distribution. The subsequent
   *     call to `grad()` will compute gradients with respect to this Random
   *     object, as the proposed value. On the other hand, `value()` causes
   *     Random objects to be subsequently treated as constant values, and
   *     a subsequent call to `grad()` will not compute gradients with
   *     respect to these.
   *
   *   * Each call to `pilot()` *must* be matched with a subsequent call to
   *     `grad()`. This is because subexpressions may occur multiple times in
   *     the same expression, and the `pilot()` call is used to determine how
   *     many times. The computations for `grad()` are then optimized by
   *     accumulating all upstream gradients before recursing into the
   *     subexpression.
   */
  final function pilot() -> Value {
    count <- count + 1;
    if !x? {
      doPilot();
      assert x?;
    }
    return x!;
  }
  
  /**
   * Evaluate before `grad()`; overridden by derived classes.
   */
  abstract function doPilot();

  /**
   * Propose a value.
   *
   * - x: The value.
   */
  final function setValue(x:Value) {
    assert !this.x?;
    this.x <- x;
    doSetValue();
  }
  
  /**
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
   */
  final function setPilot(x:Value) {
    assert !this.x?;
    this.x <- x;
    doSetPilot();
  }
  
  /**
   * Propose a pilot value; overridden by derived classes if supported
   * (notably Random).
   */
  function doSetPilot() {
    assert false;
  }

  /**
   * Evaluate gradient with respect to all Random nodes.
   *
   * - d: Upstream gradient. For an initial call, this should be the unit for
   *     the given type, e.g. 1.0, a vector of ones, or the identity matrix.
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
  function grad(d:Value) {
    assert count > 0;
    
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
    }
  }
  
  /**
   * Evaluate gradient; overriden by derived classes;
   */
  abstract function doGrad(d:Value);

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
