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
    x <- filter.x[ancestor(filter.w)].m;
    w <- filter.lnormalize;
  }
}
