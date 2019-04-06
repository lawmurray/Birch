/**
 * Bootstrap particle filter.
 */
class BootstrapParticleFilter < ParticleFilter {
  function start() {
    parallel for auto n in 1..N {
      x[n].getHandler().setMode(PLAY_IMMEDIATE);
    }
    super.start();
  }
}
