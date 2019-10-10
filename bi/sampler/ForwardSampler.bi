/**
 * Sampler for a ForwardModel archetype.
 */
class ForwardSampler < Sampler {
  /**
   * The archetype.
   */
  archetype:ForwardModel?;

  /**
   * Number of steps.
   */
  T:Integer <- 1;
  
  /**
  * Offset.
  */
  offset:Integer <- 0;
  
  function setArchetype(archetype:Model) {
    this.archetype <-? ForwardModel?(archetype);
    if !this.archetype? {
      error("model class must be a subtype of ForwardModel to use ForwardSampler.");
    }
    T <- this.archetype!.size();
  }

  function read(buffer:Buffer) {
    super.read(buffer);
    T <-? buffer.get("nsteps", T);
    offset <-? buffer.get("offset", offset);

  }

  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("nsteps", T);
    buffer.set("offset", offset);    
  }
}
