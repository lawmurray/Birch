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
      if evt.isFactor() {
        w <- w + evt.observe();
      } else if evt.isRandom() {
        if evt.hasValue() {
          w <- w + evt.observe();
        } else {
          evt.assume();
        }
      }
    }
    if verbose {
      stderr.print("log weight: " + w + "\n");
    }
    return (x, w);
  }
}
