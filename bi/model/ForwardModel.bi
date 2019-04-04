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
   * Take one step.
   */
  function play() -> Real;

  /**
   * Move forward one step.
   */
  function next();

  /**
   * Number of steps.
   */
  function size() -> Integer;
}
