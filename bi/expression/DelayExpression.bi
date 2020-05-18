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

  /**
   * Finalize contribution to the log-acceptance probability for the
   * proposed and current states. This object is considered the proposed
   * state $x^\prime$.
   *
   * - x: Current state $x$. This must be an expression of the same structure
   *      as this ($x^\prime$) but with potentially different values.
   * - κ: Markov kernel.
   *
   * Returns: contribution to the log-acceptance probability, as required for
   * the particular kernel.
   */
  final function zip(x:DelayExpression, κ:Kernel) -> Real {
    if !flagValue && !flagZip {
      flagZip < true;
      return doZip(x, κ);
    } else {
      return 0.0;
    }
  }

  /*
   * Finalize contribution to the log-acceptance probability for the
   * proposed and current states.
   */
  abstract function doZip(x:DelayExpression, κ:Kernel) -> Real;

  /**
   * Clear zip flag. If multiple calls to `zip()` are required, call this
   * in between to reset the flag used to short-circuit visits to shared
   * subexpressions.
   */
  final function clearZip() {
    if flagZip {
      flagZip <- false;
      doClearZip();
    }
  }

  /*
   * Clear zip flag.
   */
  abstract function doClearZip();
}
