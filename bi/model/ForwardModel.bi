/**
 * Forward model. To simulate the model, first use `start()`, followed by
 * `step()` any number of times.
 */
class ForwardModel < Model {   
  /**
   * Start.
   */
  function start() -> Real;

  /**
   * Step.
   */
  function step() -> Real;
  
  /**
   * Rewind to start.
   */
  function rewind();

  /**
   * Number of steps.
   */
  function size() -> Integer;

  /**
   * Update the model with parameters proposed using another model
   *
   *  - x: Previous model
   *
   * Returns: a tuple giving the proposal weight of the previous model given
   * the new model and of the new model given the previous model.
   */
  function propose(x:ForwardModel) -> (Real, Real);
}
