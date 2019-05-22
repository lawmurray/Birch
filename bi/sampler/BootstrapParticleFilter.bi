/**
 * Bootstrap particle filter.
 */
class BootstrapParticleFilter < ParticleFilter {
  function start() {
    for auto n in 1..N {
      x[n].h.setMode(PLAY_IMMEDIATE);
    }
    super.start();
  }
}
