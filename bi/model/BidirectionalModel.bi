/**
 * Bidirectional model.
 */
abstract class BidirectionalModel < ForwardModel {  
  /**
   * Move back one step.
   */
  function previous() {
    t <- t - 1;
  }
}
