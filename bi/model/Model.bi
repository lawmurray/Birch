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
   * Simulate the model to termination, yielding events.
   */
  fiber simulate() -> Event;

  /**
   * Number of steps, if known in advance.
   */
  function size() -> Integer? {
    return nil;
  }
}
