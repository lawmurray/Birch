/**
 * Abstract particle sampler that requires a conditional particle filter.
 *
 * The ParticleSampler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Sampler.svg"></object>
 * </center>
 */
abstract class ConditionalParticleSampler < ParticleSampler {
  final function sample(filter:ParticleFilter, archetype:Model) {
    auto conditionalFilter <- ConditionalParticleFilter?(filter);
    if conditionalFilter? {
      sample(conditionalFilter!, archetype);
    } else {
      error(getClassName() + " requires ConditionalParticleFilter.");
    }
  }

  final function sample(filter:ParticleFilter, archetype:Model,
      t:Integer) {
    auto conditionalFilter <- ConditionalParticleFilter?(filter);
    if conditionalFilter? {
      sample(conditionalFilter!, archetype, t);
    } else {
      error(getClassName() + " requires ConditionalParticleFilter.");
    }
  }
  
  /**
   * Conditional version of `sample(...)`.
   */
  abstract function sample(filter:ConditionalParticleFilter,
      archetype:Model);

  /**
   * Conditional version of `sample(...)`.
   */
  abstract function sample(filter:ConditionalParticleFilter,
      archetype:Model, t:Integer);
}
