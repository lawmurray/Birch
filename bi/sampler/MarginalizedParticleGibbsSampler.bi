/**
 * Marginalized particle Gibbs sampler.
 * 
 * The ParticleSampler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Sampler.svg"></object>
 * </center>
 */
class MarginalizedParticleGibbsSampler < ParticleSampler {
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
    r:Trace?;
    
    for n in 1..nsamples {
      auto f <- filter.filter(model, r);
      for t in 1..nsteps + 1 {
        f?;
        (x, w, lnormalize[t], ess[t], npropagations[t]) <- f!;
      }
      assert !r? || r!.empty();
      auto b <- ancestor(w);
      yield (x[b], 0.0, lnormalize, ess, npropagations);
      r <- x[b].trace;
    }
  }
}
