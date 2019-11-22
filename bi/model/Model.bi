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
   * Simulate.
   */
  fiber simulate(t:Integer) -> Event {
    //
  }
  
  /**
   * Forecast.
   */
  fiber forecast(t:Integer) -> Event {
    error(getClassName() + " does not support forecast.");
  }
  
  /**
   * Trace.
   */
  trace:Trace;
}
