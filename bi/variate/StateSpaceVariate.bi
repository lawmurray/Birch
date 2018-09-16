/**
 * Variate for StateSpaceModel.
 */
class StateSpaceVariate<ParameterVariate,StateVariate,ObservationVariate> {  
  /**
   * Parameter.
   */
  θ:ParameterVariate;

  /**
   * States.
   */
  x:List<StateVariate>;
  
  /**
   * Observations.
   */
  y:List<ObservationVariate>;

  function read(reader:Reader) {
    θ.read(reader.getObject("θ"));
    x.read(reader.getObject("x"));
    y.read(reader.getObject("y"));
  }
  
  function write(writer:Writer) {
    θ.write(writer.setObject("θ"));
    x.write(writer.setObject("x"));
    y.write(writer.setObject("y"));
  }
}
