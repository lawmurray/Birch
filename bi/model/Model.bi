/**
 * Base class for all models. Subclasses typically override at least the
 * `simulate` member fiber to encode the joint distribution of the model.
 * This fiber yields incrementa log weights each time a checkpoint is
 * encountered. In many cases these checkpoints are implied by the use of
 * the `~` operator, but they may also be denoted explicitly with `yield`
 * statements. 
 */
class Model {
  /**
   * Simulate the state of the model, yielding log-weights.
   *
   * Yields: a log-weight each time a checkpoint is encountered.
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
}
