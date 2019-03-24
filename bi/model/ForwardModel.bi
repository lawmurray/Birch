/**
 * Forward model. To simulate the model, first use `start()`, followed by
 * `step()` any number of times.
 */
class ForwardModel < Model {
  /**
   * Start.
   */
  fiber start() -> Event;

  /**
   * Take one step.
   */
  fiber step() -> Event;

  /**
   * Skip one step. This does not necessarily preserve a consistent state
   * state. A typical use is for an inference method that seeks to replay
   * a model in order to update its state.
   */
  function skip();

  /**
   * Number of steps.
   */
  function size() -> Integer;
}
