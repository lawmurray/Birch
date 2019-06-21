/**
 * Bidirectional model.
 */
class BidirectionalModel < ForwardModel {  
  /**
   * Move back one step.
   */
  function previous() {
    t <- t - 1;
  }
}
