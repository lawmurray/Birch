/**
 * Sampler for a ForwardModel archetype.
 */
class ForwardSampler < Sampler {
  /**
   * The archetype.
   */
  archetype:ForwardModel?;
  
  function setArchetype(archetype:Model) {
    this.archetype <-? ForwardModel?(archetype);
    if !this.archetype? {
      error("model class must be a subtype of ForwardModel to use ForwardSampler.");
    }
  }
}
