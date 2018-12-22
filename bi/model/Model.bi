/**
 * Base class for all models.
 */
class Model {
  /**
   * Simulate.
   */
  fiber simulate() -> Real {
    assert false;
  }
  
  /**
   * Get the natural number of checkpoints for this model as configured, if
   * this can be known in advance.
   */
  function checkpoints() -> Integer? {
    return nil;
  }
}
