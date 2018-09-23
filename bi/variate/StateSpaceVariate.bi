/**
 * Variate used by StateSpaceModel.
 */
class StateSpaceVariate<Parameter,State,Observation> < Variate {  
  /**
   * Parameter.
   */
  θ:Parameter;

  /**
   * States.
   */
  x:List<State>;
  
  /**
   * Observations.
   */
  y:List<Observation>;

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
