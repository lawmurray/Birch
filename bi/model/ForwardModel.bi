/**
 * Forward model. To simulate the model, first use `start()`, followed by
 * `step()` any number of times.
 */
class ForwardModel < Model {
  /**
   * Event handler.
   */
  h:Handler <- DelayHandler();
 
  /**
   * Start.
   */
  function start() -> Real;

  /**
   * Take one step.
   */
  function step() -> Real;

  /**
   * Number of steps.
   */
  function size() -> Integer;

  /**
   * Skip one step. This does not necessarily preserve a consistent state
   * state. A typical use is for an inference method that seeks to replay
   * a model in order to update its state.
   */
  function skip();
}
