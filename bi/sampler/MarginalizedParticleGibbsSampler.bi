/**
 * Marginalized particle Gibbs sampler.
 */
class MarginalizedParticleGibbsSampler < Sampler {
  /**
   * Conditional particle filter to use for state sampling.
   */
  filter:ConditionalParticleFilter;
  
  /**
   * Number of steps.
   */
  nsteps:Integer <- 0;

  fiber sample(model:ForwardModel) -> (Model, Real) {
    x:ForwardModel[_];
    w:Real[_];
    r:Trace[_];
    ess:Real;
    levidence:Real;

    /* first sample, using unconditional particle filter */    
    auto f <- filter.filter(model, nil);
    for t in 0..nsteps {
      f?;
      (x, w, r, ess, levidence) <- f!;
    }
    auto b <- ancestor(w);
    yield (x[b], 0.0);
        
    /* subsequent samples using conditional particle filter */
    while true {
      f <- filter.filter(model, r[b]);
      for t in 0..nsteps {
        f?;
        (x, w, r, ess, levidence) <- f!;
      }
      b <- ancestor(w);
      yield (x[b], 0.0);
    }
  }

  function read(buffer:Buffer) {
    nsteps <-? buffer.get("nsteps", nsteps);
  }

  function write(buffer:Buffer) {
    buffer.set("nsteps", nsteps);
  }
}
