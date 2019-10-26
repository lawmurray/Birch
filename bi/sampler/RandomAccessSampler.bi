/**
 * Sampler for a RandomAccessModel archetype.
 */
abstract class RandomAccessSampler < Sampler {
  /**
   * The archetype.
   */
  archetype:RandomAccessModel?;
  
  function setArchetype(archetype:Model) {
    this.archetype <-? RandomAccessModel?(archetype);
    if !this.archetype? {
      error("model class must be a subtype of RandomAccessModel to use RandomAccessSampler.");
    }
  }
}
