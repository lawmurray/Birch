/**
 * Base class for all models.
 */
class Model {
  /**
   * Fiber handle for incremental simulation.
   */
  f:Real!;

  /**
   * Simulate.
   */
  fiber simulate() -> Real {
    //
  }
  
  /**
   * Get the natural number of checkpoints for this model as configured, if
   * this can be known in advance.
   */
  function checkpoints() -> Integer? {
    return nil;
  }

  /**
   * Start incremental simulation. This is an alternative interface to the
   * `simulate()` fiber. Use `start()` to begin an incremental simulation,
   * returning the first weight. Use `step()` repeatedly to continue the
   * simulation, returning each additional weight.
   */
  function start() -> Real {
    f <- simulate();
    return step();
  }
  
  /**
   * Continue incremental simulation.
   */
  function step() -> Real {
    if (f?) {
      return f!;
    } else {
      return 0.0;
    }
  }
}
