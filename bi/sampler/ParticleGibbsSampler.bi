/**
 * Particle Gibbs sampler.
 */
class ParticleGibbsSampler < Sampler {
  /**
   * Conditional particle filter to use for state sampling.
   */
  filter:ConditionalParticleFilter;
  
  /**
   * Number of steps.
   */
  nsteps:Integer <- 0;

  fiber sample(model:Model) -> (Model, Real) {
    r:Trace?;
    while true {
      /* Gibbs update of parameters */
      auto x <- clone<Model>(model);
      if r? {
        auto x' <- clone<Model>(model);
        auto r' <- clone<Trace>(r!);
        auto w' <- delay.handle(x'.simulate(), r');
        for t in 1..nsteps {
          w' <- w' + redelay.handle(r', x'.simulate(t));
        }
        replay.handle(r', x.simulate());
      } else {
        play.handle(x.simulate());
      }

      /* filter */
      auto f <- filter.filter(x, r);
      for t in 0..nsteps {
        f?;
      }
      assert !r? || r!.empty();
      
      y:Model[_];
      w:Real[_];
      W:Real;
      (y, w, W) <- f!;
      auto b <- ancestor(w);
      yield (y[b], 0.0);
      r <- y[b].trace;
    }
  }

  function read(buffer:Buffer) {
    filter <-? ConditionalParticleFilter?(buffer.get("filter", filter));
    nsteps <-? buffer.get("nsteps", nsteps);
  }

  function write(buffer:Buffer) {
    buffer.set("filter", filter);
    buffer.set("nsteps", nsteps);
  }
}
