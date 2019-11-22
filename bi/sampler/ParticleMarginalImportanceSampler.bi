/**
 * Particle marginal importance sampler.
 */
class ParticleMarginalImportanceSampler < ParticleSampler {
  /**
   * Particle filter to use for state sampling.
   */
  filter:ParticleFilter;
  
  /**
   * Number of steps.
   */
  nsteps:Integer <- 0;

  fiber sample(model:Model) -> (Model, Real, Real[_], Real[_], Integer[_]) {
    x:Model[_];
    w:Real[_];
    lnormalizer:Real[nsteps + 1];
    ess:Real[nsteps + 1];
    npropagations:Integer[nsteps + 1];
    
    while true {
      auto f <- filter.filter(model);
      for t in 1..nsteps + 1 {
        f?;
        (x, w, lnormalizer[t], ess[t], npropagations[t]) <- f!;
      }
      auto b <- ancestor(w);
      yield (x[b], 0.0, lnormalizer, ess, npropagations);
    }
  }

  function read(buffer:Buffer) {
    filter <-? ParticleFilter?(buffer.get("filter", filter));
    nsteps <-? buffer.get("nsteps", nsteps);
  }

  function write(buffer:Buffer) {
    buffer.set("filter", filter);
    buffer.set("nsteps", nsteps);
  }
}
