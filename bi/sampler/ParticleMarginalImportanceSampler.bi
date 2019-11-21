/**
 * Particle marginal importance sampler.
 */
class ParticleMarginalImportanceSampler < Sampler {
  /**
   * Particle filter to use for state sampling.
   */
  filter:ParticleFilter;
  
  /**
   * Number of steps.
   */
  nsteps:Integer <- 0;

  fiber sample(model:Model) -> (Model, Real) {
    while true {
      auto f <- filter.filter(model);
      for t in 0..nsteps {
        f?;
      }
      
      x:Model[_];
      w:Real[_];
      W:Real;
      (x, w, W) <- f!;
      auto b <- ancestor(w);
      yield (x[b], W);
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
