/**
 * Abstract model.
 */
class Model {
  /**
   * Simulate.
   */
  fiber simulate() -> Real! {
    //
  }
  
  /**
   * Read inputs.
   */
  function input(reader:Reader) {
    //
  }
  
  /**
   * Write outputs.
   */
  function output(writer:Writer) {
    //
  }
  
  /**
   * Weight.
   */
  w:Real <- 0.0;
}
