/**
 * Variate used by MarkovChainModel.
 */
class MarkovChainVariate<State> < Variate {  
  /**
   * States.
   */
  x:List<State>;

  function read(reader:Reader) {
    auto f <- reader.getArray("x");
    while (f?) {
      x':State;
      x'.read(f!);
      x.pushBack(x');
    }
  }
  
  function write(writer:Writer) {    
    //writer.setArray("x", x.walk());
  }
}
