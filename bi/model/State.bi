/**
 * State concept.
 */
class State {
  /**
   * Initial model.
   *
   * - θ: Parameter.
   */
  fiber initial(θ:Parameter) -> Real;

  /**
   * Transition model.
   *
   * - x: Previous state.
   * - θ: Parameter.
   */
  fiber transition(x:State, θ:Parameter) -> Real;
  
  /**
   * Input.
   */
  function input(reader:Reader) {
    //
  }
  
  /**
   * Output.
   */
  function output(writer:Writer) {
    //
  }
}
