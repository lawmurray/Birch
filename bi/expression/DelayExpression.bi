/**
 * Value-agnostic interface for expressions.
 *
 * - isConstant: Is this a constant expression?
 */
abstract class DelayExpression(isConstant:Boolean) {
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
   * Make this a constant expression.
   */
  abstract function makeConstant();

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

  /*
   * Construct a lazy expression for the log-prior; overridden by derived
   * classes.
   */
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
