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
    auto r <- reader.getObject("θ");
    if (r?) {
      θ.read(r!);
    }
  
    auto f <- reader.getArray("x");
    while (f?) {
      x1:StateVariate;
      x1.read(f!);
      x.pushBack(x1);
    }
      
    f <- reader.getArray("y");
    while (f?) {
      y1:ObservationVariate;
      f!.read(y1);
      y.pushBack(y1);
    }
  }
  
  function write(writer:Writer) {
    //writer.setArray("x").write(x.walk());
    //writer.setArray("y").write(y.walk());
  }
}
