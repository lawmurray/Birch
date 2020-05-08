/**
 * Particle Gibbs sampler.
 * 
 * The ParticleSampler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Sampler.svg"></object>
 * </center>
 */
class ParticleGibbsSampler < ConditionalParticleSampler {
  override function sample(filter:ConditionalParticleFilter,
      archetype:Model) {
    filter.alreadyInitialized <- true;
  }
  
  override function sample(filter:ConditionalParticleFilter, archetype:Model,
      n:Integer) {
    clearDiagnostics();

    if filter.r? {
      /* Gibbs update of parameters */ 
      r:Trace <- filter.r!;       
      r':Trace;
      
      auto play <- PlayHandler(true);
      auto x' <- clone(archetype);
      auto w' <- play.handle(x'.simulate(), r');
      play <- PlayHandler(filter.delayed);
      for t in 1..filter.size() {
        w' <- w' + play.handle(filter.r!, x'.simulate(t));
      }
      
      x' <- clone(archetype);
      lnormalize.pushBack(play.handle(r', x'.simulate()));
      ess.pushBack(1.0);
      npropagations.pushBack(1);    
      filter.r!.rewind();
    }

    for t in 0..filter.size() {
      if t == 0 {
        filter.filter(archetype);
      } else {
        filter.filter(archetype, t);
      }
      pushDiagnostics(filter);
    }

    /* draw a single sample and weight with normalizing constant estimate */
    filter.b <- ancestor(filter.w);
    x <- filter.x[filter.b].m;
    w <- 0.0;
  }
}
