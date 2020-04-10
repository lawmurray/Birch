/**
 * Model.
 *
 * The Model class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Model.svg"></object>
 * </center>
 */
class Model {
  /**
   * Trace.
   */
  trace:Trace;

  /**
   * Size. This is the number of steps of `simulate(Integer)` to be performed
   * after the initial call to `simulate()`.
   */
  function size() -> Integer {
    return 0;
  }

  /**
   * Simulate.
   */
  fiber simulate() -> Event {
    //
  }

  /**
   * Simulate the `t`th step.
   */
  fiber simulate(t:Integer) -> Event {
    //
  }
}
