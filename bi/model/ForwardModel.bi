/**
 * Forward model. To simulate the model, first use `start()`, followed by
 * `play()` any number of times.
 */
class ForwardModel < Model {   
  /**
   * Start.
   */
  function start() -> Real;

  /**
   * Play one step.
   */
  function play() -> Real;
  
  /**
   * Rewind to start.
   */
  function rewind();

  /**
   * Number of steps.
   */
  function size() -> Integer;
}
