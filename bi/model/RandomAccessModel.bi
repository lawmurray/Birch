/**
 * Random access model.
 */
abstract class RandomAccessModel < BidirectionalModel {
  /**
   * Seek to a specific step.
   */
  function seek(t:Integer) {
    this.t <- t;
  }
}
