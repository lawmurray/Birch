/**
 * Particle marginal importance sampler.
 */
class ParticleMarginalImportanceSampler < ParticleSampler {
  /**
   * Particle filter to use for state sampling.
   */
  filter:ParticleFilter;
  
  fiber sample(model:Model) -> (Model, Real, Real[_], Real[_], Integer[_]) {
    /* number of steps */
    auto nsteps <- model.size();
    if filter.nsteps? {
      nsteps <- filter.nsteps!;
    }

    x:Model[_];
    w:Real[_];
    lnormalize:Real[nsteps + 1];
    ess:Real[nsteps + 1];
    npropagations:Integer[nsteps + 1];
    
    for n in 1..nsamples {
      auto f <- filter.filter(model);
      for t in 1..nsteps + 1 {
        f?;
        (x, w, lnormalize[t], ess[t], npropagations[t]) <- f!;
      }
      auto b <- ancestor(w);
      yield (x[b], 0.0, lnormalize, ess, npropagations);
    }
  }

  function read(buffer:Buffer) {
    super.read(buffer);
    filter <-? ParticleFilter?(make(buffer.getObject("filter")));
  }

  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("filter", filter);
  }
}
