/**
 * Marginalized particle Gibbs sampler.
 * 
 * The ParticleSampler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Sampler.svg"></object>
 * </center>
 */
class MarginalizedParticleGibbsSampler < ConditionalParticleSampler {
  override function sample(filter:ConditionalParticleFilter,
      archetype:Model) {
    filter.alreadyInitialized <- false;
  }

  override function sample(filter:ConditionalParticleFilter,
      archetype:Model, n:Integer) {
    clearDiagnostics();
    filter.filter(archetype);
    pushDiagnostics(filter);
    for t in 1..filter.size() {
      filter.filter(archetype, t);
      pushDiagnostics(filter);
    }
    
    /* draw a single sample and weight with normalizing constant estimate */
    x <- filter.x[ancestor(filter.w)];
    w <- filter.lnormalize;
  }
}
