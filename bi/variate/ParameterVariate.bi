/**
 * Variate for ParameterizedModel.
 */
class ParameterVariate<Parameter> < AbstractVariate {  

  function read(reader:Reader) {
    auto r <- reader.getObject("θ");
    if (r?) {
      r!.read(θ);
    }
  }
  
  function write(writer:Writer) {
    writer.setObject("θ").write(θ);
  }
}
