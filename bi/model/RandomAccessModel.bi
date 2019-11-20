/**
 * Random access model.
 */
abstract class RandomAccessModel < BidirectionalModel {
  /**
   * Simulate one step.
   *
   * - t: The step index. The caller guarantees that this is a value between
   *      1 and `size()`, which has not been used in a previous call to
   *      the same fiber.
   */
  fiber simulate(t:Integer) -> Event {
    //
  }
}
