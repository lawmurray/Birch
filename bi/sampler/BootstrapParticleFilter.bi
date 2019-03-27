/**
 * Bootstrap particle filter.
 */
class BootstrapParticleFilter < ParticleFilter {
  function setArchetype(a:Model) {
    super.setArchetype(a);
    archetype!.getHandler().setDelay(false);
  }
}
