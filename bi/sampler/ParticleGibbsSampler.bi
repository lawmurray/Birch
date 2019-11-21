/**
 * Particle Gibbs sampler.
 */
class ParticleGibbsSampler < ForwardSampler {
  /**
   * Conditional particle filter to use for state sampling.
   */
  filter:ConditionalParticleFilter;
  
  /**
   * Number of steps.
   */
  nsteps:Integer <- 0;

  fiber sample(model:ForwardModel) -> (ForwardModel, Real) {
    x:ForwardModel[_];
    w:Real[_];
    r:Trace[_];
    ess:Real;
    levidence:Real;

    /* initialize parameters */
    auto x' <- clone<ForwardModel>(model);
    auto w' <- play.handle(x'.simulate());

    /* first state sample */
    auto f <- filter.filter(x', nil);
    for t in 0..nsteps {
      f?;
      (x, w, r, ess, levidence) <- f!;
    }
    auto b <- ancestor(w);
    yield (x[b], 0.0);
        
    /* subsequent samples */
    while true {
      /* Gibbs update of parameters */
      auto x' <- clone<ForwardModel>(model);
      auto r' <- clone<Trace>(r[b]);
      auto w' <- delay.handle(x'.simulate(), r');
      for t in 1..nsteps {
        w' <- w' + redelay.handle(r', x'.simulate(t), r');
      }
      x' <- clone<ForwardModel>(model);
      w' <- replay.handle(r', x'.simulate(), r');

      /* filter state */
      f <- filter.filter(x', r');
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
