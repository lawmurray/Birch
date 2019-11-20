/**
 * Bootstrap particle filter.
 */
class BootstrapParticleFilter < ParticleFilter {
  function start() {
    for n in 1..N {
      x[n].h.setMode(PLAY_IMMEDIATE);
    }
    super.start();
  }
}
