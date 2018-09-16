/**
 * Variate for MarkovChain.
 */
class MarkovChainVariate<StateVariate> {  
  /**
   * States.
   */
  x:List<StateVariate>;

  function read(reader:Reader) {
    auto f <- reader.getArray("x");
    while (f?) {
      x':StateVariate;
      x'.read(f!);
      x.pushBack(x');
    }
  }
  
  function write(writer:Writer?) {    
    //writer.setArray("x", x.walk());
  }
}
