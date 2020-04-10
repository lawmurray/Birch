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
      
      auto x' <- clone(archetype);
      auto w' <- playDelay.handle(x'.simulate(), r');
      for t in 1..filter.size() {
        w' <- w' + replay.handle(filter.r!, x'.simulate(t));
      }
      
      x' <- clone(archetype);
      lnormalize.pushBack(replay.handle(r', x'.simulate()));
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
  }
}
