/**
 * Particle marginal importance sampler.
 */
class ParticleMarginalImportanceSampler < ParticleSampler {
  /**
   * Particle filter to use for state sampling.
   */
  filter:ParticleFilter;
  
  fiber sample(model:Model) -> (Model, Real, Real[_], Real[_], Integer[_]) {
    auto nsteps <- filter.nsteps;
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
  }

  function write(buffer:Buffer) {
    buffer.set("filter", filter);
  }
}
