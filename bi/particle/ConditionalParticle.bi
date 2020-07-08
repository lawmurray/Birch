/**
 * Particle for use with ConditionalParticleFilter.
 *
 * - m: Model.
 *
 * The Particle class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Particle.svg"></object>
 * </center>
 */
class ConditionalParticle(m:Model) < Particle(m) {
  /**
   * Tape<Record> of the model simulation. This is required in order to replay the
   * particle.
   */
  trace:Tape<Record>;
}

/**
 * Create a ConditionalParticle.
 */
function ConditionalParticle(m:Model) -> ConditionalParticle {
  return construct<ConditionalParticle>(m);
}
