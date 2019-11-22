/**
 * Particle Gibbs sampler.
 */
class ParticleGibbsSampler < ParticleSampler {
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

    while true {
      /* Gibbs update of parameters */
      auto m <- clone<Model>(model);
      if r? {
        auto x' <- clone<Model>(model);
        auto r' <- clone<Trace>(r!);
        auto w' <- delay.handle(x'.simulate(), r');
        for t in 1..nsteps {
          w' <- w' + redelay.handle(r', x'.simulate(t));
        }
        replay.handle(r', m.simulate());
      } else {
        play.handle(m.simulate());
      }

      /* filter */
      auto f <- filter.filter(m, r);
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
    filter <-? ConditionalParticleFilter?(buffer.get("filter", filter));
  }

  function write(buffer:Buffer) {
    buffer.set("filter", filter);
  }
}
