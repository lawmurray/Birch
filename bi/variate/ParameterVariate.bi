/**
 * Variate for ParameterizedModel.
 */
class ParameterVariate<Parameter> < AbstractVariate {  

  function read(reader:Reader) {
    θ.read(reader.getObject("θ"));
  }
  
  function write(writer:Writer?) {
    θ.write(writer.setObject("θ"));
  }
}
