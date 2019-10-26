/**
 * Forward model. To simulate the model, first use `start()`, followed by
 * `step()` any number of times.
 */
abstract class ForwardModel < Model {   
  /**
   * Current step.
   */
  t:Integer <- 0;
  
  /**
   * Start.
   */
  function start() -> Real {
    return 0.0;
  }

  /**
   * Step.
   */
  function step() -> Real {
    t <- t + 1;
    return 0.0;
  }
  
  /**
   * Rewind to start.
   */
  function rewind() {
    t <- 0;
  }

  /**
   * Number of steps.
   */
  abstract function size() -> Integer;

  /**
   * Update the model with parameters proposed using another model
   *
   *  - x: Previous model
   *
   * Returns: a tuple giving the proposal weight of the previous model given
   * the new model and of the new model given the previous model.
   */
  abstract function propose(x:ForwardModel) -> (Real, Real);
}
