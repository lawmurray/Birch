/**
 * State concept.
 */
class State {
  /**
   * Simulate initial state.
   *
   * - θ: Parameter.
   */
  fiber simulate(θ:Model) -> Real!;

  /**
   * Simulate transition.
   *
   * - x: Previous state.
   * - θ: Parameter.
   */
  fiber simulate(x:State, θ:Model) -> Real!;
  
  /**
   * Input to the state.
   */
  function input(reader:Reader);
  
  /**
   * Output from the state.
   */
  function output(writer:Writer);
}
