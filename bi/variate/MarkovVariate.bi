/**
 * Variate used by MarkovModel.
 */
class MarkovVariate<Parameter,State> < Variate {  
  /**
   * States.
   */
  x:List<State>;

  function read(reader:Reader) {
    x.read(reader.getObject("x"));
  }
  
  function write(writer:Writer) {    
    x.write(writer.setObject("x"));
  }
}
