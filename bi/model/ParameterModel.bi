/**
 * Abstract model with a parameter.
 */
class ParameterModel<Parameter> < Model {
  /**
   * Parameter.
   */
  θ:Parameter;

  /**
   * Parameter model.
   *
   * - θ: The parameters, to be set.
   */
  fiber parameter(θ:Parameter) -> Real {
    //
  }

  function read(buffer:Buffer) {
    super.read(buffer);
    buffer.get("θ", θ);
  }
  
  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("θ", θ);
  }
}
