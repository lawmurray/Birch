/**
 * Abstract model.
 *
 * The Model class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Model.svg"></object>
 * </center>
 */
class Model {
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
   * Trace.
   */
  trace:Trace;
}
