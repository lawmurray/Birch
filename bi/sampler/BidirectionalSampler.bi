/**
 * Sampler for a BidirectionalModel archetype.
 */
class BidirectionalSampler < Sampler {
  /**
   * The archetype.
   */
  archetype:BidirectionalModel?;
  
  function setArchetype(archetype:Model) {
    this.archetype <-? BidirectionalModel?(archetype);
    if !this.archetype? {
      error("model class must be a subtype of BidirectionalModel to use BidirectionalSampler.");
    }
  }
}
