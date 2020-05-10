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
 * Value-agnostic interface for expressions. Provides essential
 * functionality for Markov kernels.
 */
abstract class DelayExpression {
  /**
   * Expression state.
   */
  state:ExpressionState <- EXPRESSION_INITIAL;
  
  /**
   * Construct a lazy expression for the log-prior.
   *
   * Pre-condition: the expression is in the pilot or gradient state.
   *
   * Returns: If the expression is a variable, the log-prior expression,
   * otherwise nil, which may be interpreted as the log-prior evaluating to
   * zero.
   */
  final function prior() -> Expression<Real>? {
    assert state == EXPRESSION_PILOT || state == EXPRESSION_GRADIENT;
    return doPrior();
  }

  /*
   * Construct a lazy expression for the log-prior; overridden by derived
   * classes.
   */
  function doPrior() -> Expression<Real>? {
    return nil;
  }
  
  /**
   * Evaluate log-ratio of proposal densities.
   *
   * Pre-condition: the expression is in the pilot or gradient state.
   *
   * Returns: the log-prior expression, or nil if no such Random objectss
   * exist, which may be interpreted as the log-prior evaluating to zero.
   */
  final function ratio(to:DelayExpression) -> Real {
    assert state == EXPRESSION_PILOT || state == EXPRESSION_GRADIENT;
    return doRatio(to);
  }
  
  /*
   * Evaluate log-ratio of proposal densities; overridden by derived classes.
   */
  function doRatio(to:DelayExpression) -> Real {
    return 0.0;
  }
}
