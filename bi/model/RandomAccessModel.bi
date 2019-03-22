/**
 * Random access model.
 */
class RandomAccessModel < BidirectionalModel {
  /**
   * Current step.
   */
  t:Integer <- 0;

  /**
   * Seek to a specific step.
   */
  function seek(t:Integer) {
    this.t <- t;
  }
  
  function skip() {
    t <- t + 1;
  }
  
  function back() {
    t <- t - 1;
  }
}
