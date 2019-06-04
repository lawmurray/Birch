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
    
  function step() -> Real {
    seek(t + 1);
  }
  
  function previous() {
    seek(t - 1);
  }
  
  function rewind() {
    seek(0);
  }
}
