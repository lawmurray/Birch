/**
 * Abstract model with observations.
 */
class ObservedModel<Parameter,State,Observation> < StateModel<Parameter,State> {
  /**
   * Observations.
   */
  y:List<Observation>;

  function read(buffer:Buffer) {
    super.read(buffer);
    buffer.get("y", y);
  }
  
  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("y", y);
  }
}
