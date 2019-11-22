/**
 * Marginalized particle Gibbs sampler.
 */
class MarginalizedParticleGibbsSampler < ParticleSampler {
  /**
   * Conditional particle filter to use for state sampling.
   */
  filter:ConditionalParticleFilter;

  fiber sample(model:Model) -> (Model, Real, Real[_], Real[_], Integer[_]) {
    auto nsteps <- filter.nsteps;
    x:Model[_];
    w:Real[_];
    lnormalizer:Real[nsteps + 1];
    ess:Real[nsteps + 1];
    npropagations:Integer[nsteps + 1];
    r:Trace?;
    
    for n in 1..nsamples {
      auto f <- filter.filter(model, r);
      for t in 1..nsteps + 1 {
        f?;
        (x, w, lnormalizer[t], ess[t], npropagations[t]) <- f!;
      }
      assert !r? || r!.empty();
      auto b <- ancestor(w);
      yield (x[b], 0.0, lnormalizer, ess, npropagations);
      r <- x[b].trace;
    }
  }

  function read(buffer:Buffer) {
    super.read(buffer);
    filter <-? ConditionalParticleFilter?(buffer.get("filter", filter));
  }

  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("filter", filter);
  }
}
