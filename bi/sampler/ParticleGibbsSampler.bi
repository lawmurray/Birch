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

    for n in 1..nsamples {
      auto m <- clone(model);
      if n == 1 {
        /* parameter elimination as a reasonable initialization */
        lnormalize[1] <- playDelay.handle(m.simulate());
      } else {
        /* Gibbs update of parameters */        
        r':Trace;
        auto x' <- clone(model);
        auto w' <- playDelay.handle(x'.simulate(), r');
        for t in 1..nsteps {
          w' <- w' + replay.handle(r!, x'.simulate(t));
        }
        lnormalize[1] <- replay.handle(r', m.simulate());
        r!.rewind();
      }
      ess[1] <- 1.0;
      npropagations[1] <- 1;

      /* filter */
      auto f <- filter.filter(m, r, true);
      for t in 2..nsteps + 1 {
        f?;
        lnormalize':Real;
        ess':Real;
        npropagations':Integer;
        (x, w, lnormalize', ess', npropagations') <- f!;
        lnormalize[t] <- lnormalize';
        ess[t] <- ess';
        npropagations[t] <- npropagations';
      }
      
      auto b <- ancestor(w);
      yield (x[b], 0.0, lnormalize, ess, npropagations);
      r <- x[b].trace;
      r!.rewind();
    }
  }
}
