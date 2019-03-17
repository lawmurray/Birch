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

  /**
   * Parameter proposal.
   *
   * - θ: The parameters, to be set.
   *
   * By default calls `parameter(θ)`.
   */
  fiber proposeParameter(θ:Parameter) -> Real {
    parameter(θ);
  }
  
  /**
   * Parameter proposal.
   *
   * - θ': The proposed parameters, to be set.
   * - θ: The last parameters.
   *
   * By default calls `proposeParameter(θ')`.
   */
  fiber proposeParameter(θ':Parameter, θ:Parameter) -> Real {
    proposeParameter(θ');
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
