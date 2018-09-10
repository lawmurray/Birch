/**
 * Variate for StateSpaceModel.
 */
class StateSpaceVariate<Parameter,State,Observation> <
    ParameterVariate<Parameter> {  
  /**
   * States.
   */
  x:List<State>;
  
  /**
   * Observations.
   */
  y:List<Observation>;

  function read(reader:Reader) {
    super.read(reader);
  
    auto f <- reader.getArray("x");
    while (f?) {
      x1:State;
      x1.read(f!);
      x.pushBack(x1);
    }
      
    f <- reader.getArray("y");
    while (f?) {
      y1:Observation;
      f!.read(y1);
      y.pushBack(y1);
    }
  }
  
  function write(writer:Writer) {
    super.write(writer);
    
    writer.setArray("x").write(x.walk());
    writer.setArray("y").write(y.walk());
  }
}
