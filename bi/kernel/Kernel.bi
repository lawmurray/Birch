/**
 * Markov kernel.
 *
 * The Kernel class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Kernel.svg"></object>
 * </center>
 *
 * The basic use of a Kernel is to pass it to the `move()` member function of
 * an Expression to simultaneously update variables and re-evaluate the
 * expression. The Expression typically represents the log-density of a
 * target distribution, and the move represents a proposal, which will be
 * subsequently accepted or rejected (according to e.g. a
 * Metropolis--Hastings acceptance probability).
 *
 * A Kernel has three basic operations:
 *
 * - `move()`: to simulate a new value for a variable given its previous
 *   value,
 * - `logpdf()`: to evaluate the log-density of the new value for a variable
 *   given its previous value, and
 * - `zip()`: which is used for any final computations necessary to decide
 *   whether to accept or reject a move, for example computing the log-ratio
 *   of proposal densities for Metropolis--Hastings.
 *
 * For example, consider a random-walk Metropolis-Hastings algorithm
 * targeting a distribution $\pi(\mathrm{d}x)$. The Kernel object represents
 * a Markov kernel $\kappa(\mathrm{d}x' \mid x)$ ergodic and invariant to
 * $\pi(\mathrm{d}x)$. The object encodes a proposal distribution
 * $q(\mathrm{d}x^\prime \mid x)$, with the member functions `move()`
 * defining how to simulate from it, and `logpdf()` how to compute the
 * density $q(x^\prime \mid x)$. In order to accept or reject a proposal it
 * is necessary to compute the Metropolis--Hastings acceptance probability:
 *
 * $$\alpha := \min \left(1, \frac{\pi(x^\prime) q(x \mid x^\prime)}{\pi(x)
 * q(x^\prime \mid x)}\right).$$
 *
 * One approach is as follows:
 *
 * 1. An Expression object is used to evaluate $\log \pi(x)$.
 * 2. Calling `move(κ)` on the Expression object proposes a new state via
 *    subsequent calls to `κ.move(...)`, as well as computing
 *    $\log \pi(x^\prime)$.
 * 3. Calling `zip(κ)` on the Expression object computes the log-ratio
 *    $\log q(x \mid x^\prime) - \log q(x^\prime \mid x)$, via subsequent
 *    calls to `zip(...)` on the Kernel object `κ`.
 */
class Kernel {
  /**
   * Propose a transition.
   *
   * - x: Current state.
   *
   * Returns: Proposed state.
   */
  function move(x:Random<Real>) -> Real {
    return x.x!;
  }

  /**
   * Propose a transition.
   *
   * - x: Current state.
   *
   * Returns: Proposed state.
   */
  function move(x:Random<Real[_]>) -> Real[_] {
    return x.x!;
  }

  /**
   * Propose a transition.
   *
   * - x: Current state.
   *
   * Returns: Proposed state.
   */
  function move(x:Random<Real[_,_]>) -> Real[_,_] {
    return x.x!;
  }

  /**
   * Propose a transition.
   *
   * - x: Current state.
   *
   * Returns: Proposed state.
   */
  function move(x:Random<Integer>) -> Integer {
    return x.x!;
  }

  /**
   * Propose a transition.
   *
   * - x: Current state.
   *
   * Returns: Proposed state.
   */
  function move(x:Random<Integer[_]>) -> Integer[_] {
    return x.x!;
  }

  /**
   * Propose a transition.
   *
   * - x: Current state.
   *
   * Returns: Proposed state.
   */
  function move(x:Random<Boolean>) -> Boolean {
    return x.x!;
  }

  /**
   * Observe a transition.
   *
   * - x': Proposed state $x^\prime$.
   * - x: Current state $x$.
   *
   * Returns: the log probability density $q(x^\prime \mid x)$.
   */
  function logpdf(x':Random<Real>, x:Random<Real>) -> Real {
    return 0.0;
  }

  /**
   * Observe a transition.
   *
   * - x': Proposed state $x^\prime$.
   * - x: Current state $x$.
   *
   * Returns: the log probability density $q(x^\prime \mid x)$.
   */
  function logpdf(x':Random<Real[_]>, x:Random<Real[_]>) -> Real {
    return 0.0;
  }

  /**
   * Observe a transition.
   *
   * - x': Proposed state $x^\prime$.
   * - x: Current state $x$.
   *
   * Returns: the log probability density $q(x^\prime \mid x)$.
   */
  function logpdf(x':Random<Real[_,_]>, x:Random<Real[_,_]>) -> Real {
    return 0.0;
  }

  /**
   * Observe a transition.
   *
   * - x': Proposed state $x^\prime$.
   * - x: Current state $x$.
   *
   * Returns: the log probability mass $q(x^\prime \mid x)$.
   */
  function logpdf(x':Random<Integer>, x:Random<Integer>) -> Real {
    return 0.0;
  }

  /**
   * Observe a transition.
   *
   * - x': Proposed state $x^\prime$.
   * - x: Current state $x$.
   *
   * Returns: the log probability mass $q(x^\prime \mid x)$.
   */
  function logpdf(x':Random<Integer[_]>, x:Random<Integer[_]>) -> Real {
    return 0.0;
  }

  /**
   * Observe a transition.
   *
   * - x': Proposed state $x^\prime$.
   * - x: Current state $x$.
   *
   * Returns: the log probability mass.
   */
  function logpdf(x':Random<Boolean>, x:Random<Boolean>) -> Real {
    return 0.0;
  }

  /**
   * Finalize contribution to the log-acceptance probability for the
   * proposed and current states.
   *
   * - x': Proposed state $x^\prime$.
   * - x: Current state $x$.
   *
   * Returns: contribution to the log-acceptance probability, as required for
   * the particular kernel.
   */
  function zip(x':Random<Real>, x:Random<Real>) -> Real {
    return 0.0;
  }

  /**
   * Finalize contribution to the log-acceptance probability for the
   * proposed and current states.
   *
   * - x': Proposed state $x^\prime$.
   * - x: Current state $x$.
   *
   * Returns: contribution to the log-acceptance probability, as required for
   * the particular kernel.
   */
  function zip(x':Random<Real[_]>, x:Random<Real[_]>) -> Real {
    return 0.0;
  }

  /**
   * Finalize contribution to the log-acceptance probability for the
   * proposed and current states.
   *
   * - x': Proposed state $x^\prime$.
   * - x: Current state $x$.
   *
   * Returns: contribution to the log-acceptance probability, as required for
   * the particular kernel.
   */
  function zip(x':Random<Real[_,_]>, x:Random<Real[_,_]>) -> Real {
    return 0.0;
  }

  /**
   * Finalize contribution to the log-acceptance probability for the
   * proposed and current states.
   *
   * - x': Proposed state $x^\prime$.
   * - x: Current state $x$.
   *
   * Returns: contribution to the log-acceptance probability, as required for
   * the particular kernel.
   */
  function zip(x':Random<Integer>, x:Random<Integer>) -> Real {
    return 0.0;
  }

  /**
   * Finalize contribution to the log-acceptance probability for the
   * proposed and current states.
   *
   * - x': Proposed state $x^\prime$.
   * - x: Current state $x$.
   *
   * Returns: contribution to the log-acceptance probability, as required for
   * the particular kernel.
   */
  function zip(x':Random<Integer[_]>, x:Random<Integer[_]>) -> Real {
    return 0.0;
  }

  /**
   * Finalize contribution to the log-acceptance probability for the
   * proposed and current states.
   *
   * - x': Proposed state $x^\prime$.
   * - x: Current state $x$.
   *
   * Returns: contribution to the log-acceptance probability, as required for
   * the particular kernel.
   */
  function zip(x':Random<Boolean>, x:Random<Boolean>) -> Real {
    return 0.0;
  }
}
