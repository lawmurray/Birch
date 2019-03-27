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
    auto w <- x.play();    
    if verbose {
      stderr.print("log weight: " + w + "\n");
    }
    return (x, w);
  }
}
