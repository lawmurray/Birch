/**
 * Value-agnostic interface for expressions.
 *
 * - isConstant: Is this a constant expression?
 */
abstract class DelayExpression(isConstant:Boolean) {
  /**
   * Number of times `pilot()` has been called.
   */
  pilotCount:Integer16 <- 0;

  /**
   * Number of times `grad()` or `move()` has been called. In the former
   * case, used to track accumulation of upstream gradients before recursion,
   * after which it is reset to zero. In the latter case, used to ensure that
   * each subexpression is moved only once, and upon reaching `pilotCount` is
   * reset to zero.
   */
  gradCount:Integer16 <- 0;

  /**
   * Has `value()` been called? This is used as a short-circuit for shared
   * subexpressions.
   */
  flagConstant:Boolean <- isConstant;

  /**
   * Has `prior()` been called? This is used as a short-circuit for shared
   * subexpressions.
   */
  flagPrior:Boolean <- false;

  /**
   * Is this a Random expression?
   */
  function isRandom() -> Boolean {
    return false;
  }
  
  /**
   * Is this a constant expression?
   */
  function isConstant() -> Boolean {
    return flagConstant;
  }

  /**
   * Length of result. This is synonymous with `rows()`.
   */
  final function length() -> Integer {
    return rows();
  }

  /**
   * Number of rows in result.
   */
  abstract function rows() -> Integer;
  
  /**
   * Number of columns in result.
   */
  abstract function columns() -> Integer;

  /**
   * Depth of the expression tree.
   */
  abstract function depth() -> Integer;
  
  /**
   * Construct a lazy expression for the log-prior, and collect variables.
   *
   * - vars: Container into which to collect variables.
   *
   * Returns: An expression giving the log-prior of any variables in the
   * expression, or nil if there are no variables (which may be interpreted
   * as a log-prior of zero). Meanwhile, those variables are accumulated in
   * the argument `vars`, calling `pushBack()` for each.
   */
  final function prior(vars:RaggedArray<DelayExpression>) ->
      Expression<Real>? {
    if !flagPrior {
      flagPrior <- true;
      return doPrior(vars);
    } else {
      return nil;
    }
  }

  abstract function doPrior(vars:RaggedArray<DelayExpression>) ->
      Expression<Real>?;
      
  /**
   * Compute the log-pdf of a proposed state. This expression is considered
   * the current state, $x$.
   *
   * - x': Proposed state $x^\prime$.
   * - κ: Markov kernel.
   *
   * Returns: $\log q(x^\prime \mid x)$.
   *
   * This is only valid for [Random](../classes/Random/) objects. It returns
   * zero for anything else.
   */
  function logpdf(x':DelayExpression, κ:Kernel) -> Real {
    return 0.0;
  }
}

/**
 * Length of a vector.
 */
function length(x:DelayExpression) -> Integer {
  return x.length();
}

/**
 * Number of rows of a vector; equals `length()`.
 */
function rows(x:DelayExpression) -> Integer {
  return x.rows();
}

/**
 * Number of columns of a vector; equals 1.
 */
function columns(x:DelayExpression) -> Integer {
  return x.columns();
}
