/**
 * Abstract model with states.
 */
class StateModel<Parameter,State> < ParameterModel<Parameter> {
  /**
   * States.
   */
  x:List<State>;

  function read(buffer:Buffer) {
    super.read(buffer);
    buffer.get("x", x);
  }
  
  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("x", x);
  }
}
