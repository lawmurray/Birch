/**
 * Forward model.
 */
class ForwardModel < Model {   
  /**
   * Simulate one step.
   *
   * - t: The step index. The caller guarantees that this is one greater than
   *      the `t` of the previous call to the same fiber, or 1 if the fiber
   *      has not been called on this before.
   */
  fiber simulate(t:Integer) -> Event {
    //
  }
  
  /**
   * Number of steps.
   */
  function size() -> Integer {
    return 0;
  }
  
  /**
   * Read one step.
   *
   * - t: The step index.
   */
  function read(t:Integer, buffer:Buffer) {
    //
  }

  /**
   * Write one step.
   *
   * - t: The step index.
   */
  function write(t:Integer, buffer:Buffer) {
    //
  }
}
