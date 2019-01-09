/**
 * Base class for all models.
 */
class Model {
  /**
   * Fiber handle for incremental simulation.
   */
  f:Real!;

  /**
   * Start simulation.
   */
  function start() {
    f <- simulate();
  }
  
  /**
   * Step simulation to the next checkpoint.
   */
  function step() -> Real? {
    if (f?) {
      return f!;
    } else {
      return nil;
    }
  }

  /**
   * Get the natural number of checkpoints for this model as configured, if
   * this can be known in advance.
   */
  function checkpoints() -> Integer? {
    return nil;
  }

  /**
   * Simulate.
   */
  fiber simulate() -> Real {
    //
  }
  
  /**
   * Propose.
   */
  fiber propose(m:Model) -> Real {
    simulate();
  }
}
