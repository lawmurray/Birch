/**
 * Abstract model.
 *
 * The Model class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Model.svg"></object>
 * </center>
 */
abstract class Model {
  /**
   * Event handler.
   */
  h:EventHandler;

  /**
   * Play the complete model with an event handler, returning a log-weight.
   */
  function play() -> Real {
    return h.handle(simulate());
  }

  /**
   * Simulate the model, yielding events.
   */
  fiber simulate() -> Event {
    //
  }
}
