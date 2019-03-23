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

  function sample() -> (Model, Real) {
    assert archetype?;
    auto x <- clone<Model>(archetype!);
    auto f <- x.simulate();
    auto w <- 0.0;
    while f? {
      auto evt <- f!;
      if evt.isFactor() || (evt.isRandom() && evt.hasValue()) {
        w <- w + evt.observe();
      }
    }
    return (x, w);
  }
}
