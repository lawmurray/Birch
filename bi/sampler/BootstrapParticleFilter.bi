/**
 * Bootstrap particle filter.
 */
class BootstrapParticleFilter < ParticleFilter {
  function setArchetype(a:Model) {
    super.setArchetype(a);
    archetype!.setHandler(EagerHandler());
  }
}
