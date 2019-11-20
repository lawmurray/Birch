/**
 * Importance sampler.
 */
class ImportanceSampler < Sampler {
  /**
   * The archetype.
   */
  archetype:Model?;
  
  function setArchetype(archetype:Model) {
    this.archetype <- archetype;
  }

  fiber sample() -> (Model, Real) {
    assert archetype?;
    for n in 1..nsamples {
      auto x <- clone<Model>(archetype!);
      auto w <- x.play();
      yield (clone<Model>(x), w);
    }
  }
}
