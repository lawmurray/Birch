/**
 * Particle for use with ParticleFilter.
 *
 * - m: Model.
 *
 * The Particle class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Particle.svg"></object>
 * </center>
 */
class Particle(m:Model) {
  /**
   * State.
   */
  m:Model <- m;
  
  override function write(buffer:Buffer) {
    buffer.set(m);
  }
}

/**
 * Create a Particle.
 */
function Particle(m:Model) -> Particle {
  o:Particle(m);
  return o;
}
