/**
 * Marginalized particle Gibbs sampler.
 */
class MarginalizedParticleGibbsSampler < Sampler {
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
      auto f <- filter.filter(model, r);
      for t in 0..nsteps {
        f?;
      }
      assert !r? || r!.empty();
      
      x:Model[_];
      w:Real[_];
      W:Real;
      (x, w, W) <- f!;
      auto b <- ancestor(w);
      yield (x[b], 0.0);
      r <- x[b].trace;
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
