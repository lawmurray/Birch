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
  
  function setArchetype(archetype:Model) {
    this.archetype <-? ForwardModel?(archetype);
    if !this.archetype? {
      error("model class must be a subtype of ForwardModel to use ForwardSampler.");
    }
  }

  function read(buffer:Buffer) {
    super.read(buffer);
    T <-? buffer.get("nsteps", T);
  }

  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("nsteps", T);
  }
}
