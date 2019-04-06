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
   * Event handler.
   */
  h:EventHandler;

  /**
   * Get the event handler.
   */
  function getHandler() -> EventHandler {
    return h;
  }
 
  /**
   * Set the event handler.
   */
  function setHandler(h:EventHandler) {
    this.h <- h;
  }

  /**
   * Play the model with an event handler, yielding a log-weight.
   */
  function play() -> Real {
    return h.handle(simulate());
  }

  /**
   * Simulate the model, yielding events.
   */
  fiber simulate() -> Event;
}
