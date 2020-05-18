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
   * Has `zip()` been called? This is used as a short-circuit for shared
   * subexpressions. As `zip()` may need to be called multiple times, the
   * flag may be cleared with `clearZip()`.
   */
  flagZip:Boolean <- false;

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
}
