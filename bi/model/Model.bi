/**
 * Base class for all models.
 */
class Model {
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

  /**
   * Get the natural number of checkpoints for this model as configured, if
   * this can be known in advance.
   */
  function checkpoints() -> Integer? {
    return nil;
  }
}
