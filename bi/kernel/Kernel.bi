/**
 * Markov kernel.
 *
 * The Kernel class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Kernel.svg"></object>
 * </center>
 */
class Kernel {
  /**
   * Propose a transition.
   *
   * - x: Start state.
   *
   * Returns: Proposed state.
   */
  function simulate(x:Random<Real>) -> Real {
    return x.x!;
  }

  /**
   * Propose a transition.
   *
   * - x: Start state.
   *
   * Returns: Proposed state.
   */
  function simulate(x:Random<Real[_]>) -> Real[_] {
    return x.x!;
  }

  /**
   * Propose a transition.
   *
   * - x: Start state.
   *
   * Returns: Proposed state.
   */
  function simulate(x:Random<Real[_,_]>) -> Real[_,_] {
    return x.x!;
  }

  /**
   * Propose a transition.
   *
   * - x: Start state.
   *
   * Returns: Proposed state.
   */
  function simulate(x:Random<Integer>) -> Integer {
    return x.x!;
  }

  /**
   * Propose a transition.
   *
   * - x: Start state.
   *
   * Returns: Proposed state.
   */
  function simulate(x:Random<Integer[_]>) -> Integer[_] {
    return x.x!;
  }

  /**
   * Propose a transition.
   *
   * - x: Start state.
   *
   * Returns: Proposed state.
   */
  function simulate(x:Random<Boolean>) -> Boolean {
    return x.x!;
  }

  /**
   * Observe a transition.
   *
   * - x': End state.
   * - x: Start state.
   *
   * Returns: the log probability density.
   */
  function logpdf(x':Random<Real>, x:Random<Real>) -> Real {
    return 0.0;
  }

  /**
   * Observe a transition.
   *
   * - x': End state.
   * - x: Start state.
   *
   * Returns: the log probability density.
   */
  function logpdf(x':Random<Real[_]>, x:Random<Real[_]>) -> Real {
    return 0.0;
  }

  /**
   * Observe a transition.
   *
   * - x': End state.
   * - x: Start state.
   *
   * Returns: the log probability density.
   */
  function logpdf(x':Random<Real[_,_]>, x:Random<Real[_,_]>) -> Real {
    return 0.0;
  }

  /**
   * Observe a transition.
   *
   * - x': End state.
   * - x: Start state.
   *
   * Returns: the log probability mass.
   */
  function logpdf(x':Random<Integer>, x:Random<Integer>) -> Real {
    return 0.0;
  }

  /**
   * Observe a transition.
   *
   * - x': End state.
   * - x: Start state.
   *
   * Returns: the log probability mass.
   */
  function logpdf(x':Random<Integer[_]>, x:Random<Integer[_]>) -> Real {
    return 0.0;
  }

  /**
   * Observe a transition.
   *
   * - x': End state.
   * - x: Start state.
   *
   * Returns: the log probability mass.
   */
  function logpdf(x':Random<Boolean>, x:Random<Boolean>) -> Real {
    return 0.0;
  }
}
