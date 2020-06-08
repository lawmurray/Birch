/**
 * Particle marginal importance sampler.
 * 
 * The ParticleSampler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Sampler.svg"></object>
 * </center>
 */
class ParticleMarginalImportanceSampler < ParticleSampler {
  override function sample(filter:ParticleFilter, archetype:Model,
      n:Integer) {    
    clearDiagnostics();
    filter.filter(archetype);
    pushDiagnostics(filter);
    for t in 1..filter.size() {
      filter.filter(archetype, t);
      pushDiagnostics(filter);
    }

    /* draw a single sample and weight with normalizing constant estimate */
    auto b <- ancestor(filter.w);
    if b == 0 {
      warn("particle filter degenerated, problem sample will be assigned zero weight");
      w <- -inf;
    } else {
      x <- filter.x[b].m;
      w <- filter.lnormalize;
    }
  }
}
