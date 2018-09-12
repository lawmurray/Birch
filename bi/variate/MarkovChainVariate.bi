/**
 * Variate for MarkovChain.
 */
class MarkovChainVariate<StateVariate> {  
  /**
   * States.
   */
  x:List<StateVariate>;

  function read(reader:Reader) {
    auto f <- reader.getArray("θ");
    while (f?) {
      x1:StateVariate;
      x1.read(f!);
      x.pushBack(x1);
    }
  }
  
  function write(writer:Writer) {    
    //writer.setArray("x", x.walk());
  }
}
