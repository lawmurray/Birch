/**
 * Particle Gibbs sampler.
 * 
 * The ParticleSampler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Sampler.svg"></object>
 * </center>
 */
class ParticleGibbsSampler < ParticleSampler {
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
    r:Trace?;   // reference trace
    r':Trace?;  // replay trace for parameters

    for n in 1..nsamples {
      auto m <- clone<Model>(model);
      if !r? {
        /* first iteration, marginalize out parameters to obtain a reasonable
         * initialization */
        lnormalize[1] <- delay.handle(m.simulate());
      } else {
        /* second and subsequent iterations, Gibbs update of parameters */        
        auto x' <- clone<Model>(model);
        auto r' <- clone<Trace>(r!);
        auto w' <- delay.handle(x'.simulate(), r');
        for t in 1..nsteps {
          w' <- w' + replay.handle(r', x'.simulate(t));
        }
        lnormalize[1] <- replay.handle(r', m.simulate());
      }
      ess[1] <- 1.0;
      npropagations[1] <- 1;

      /* filter */
      auto f <- filter.filter(m, r, true);
      for t in 2..nsteps + 1 {
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
