/**
 * A model.
 */
class Model {  
  /**
   * Simulate the model to termination, yielding events.
   */
  fiber simulate() -> Event;

  /**
   * Number of steps, if known in advance.
   */
  function size() -> Integer? {
    return nil;
  }
}
