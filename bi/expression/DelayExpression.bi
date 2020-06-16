/**
 * Value-agnostic interface for expressions.
 */
abstract class DelayExpression {
  /**
   * Count of the number of times `pilot()` or `move()` has been called on
   * this, minus the number of times `grad()` has been called. This is used
   * to accumulate upstream gradients before recursing into a subexpression
   * that may be shared.
   */
  count:Integer <- 0;
  
  /**
   * Has `value()` been called? This is used as a short-circuit for shared
   * subexpressions.
   */
  flagValue:Boolean <- false;

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
   * For a vector, get an element.
   *
   * - i: Element index.
   */
  function element(i:Expression<Integer>) -> Expression<Real> {
    assert false;
    return box(0.0);
  }

  /**
   * For a vector, get an element.
   *
   * - i: Element index.
   */
  final function element(i:Integer) -> Expression<Real> {
    return element(box(i));
  }

  /**
   * For a matrix, get an element.
   *
   * - i: Row index.
   * - j: Column index.
   */
  function element(i:Expression<Integer>, j:Expression<Integer>) ->
      Expression<Real> {
    assert false;
    return box(0.0);
  }

  /**
   * For a matrix, get an element.
   *
   * - i: Row index.
   * - j: Column index.
   */
  function element(i:Integer, j:Expression<Integer>) -> Expression<Real> {
    return element(box(i), j);
  }

  /**
   * For a matrix, get an element.
   *
   * - i: Row index.
   * - j: Column index.
   */
  function element(i:Expression<Integer>, j:Integer) -> Expression<Real> {
    return element(i, box(j));
  }

  /**
   * For a matrix, get an element.
   *
   * - i: Row index.
   * - j: Column index.
   */
  function element(i:Integer, j:Integer) -> Expression<Real> {
    return element(box(i), box(j));
  }
  
  /**
   * Is this a constant expression?
   */
  function isConstant() -> Boolean {
    return flagValue;
  }

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
