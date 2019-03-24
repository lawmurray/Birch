/**
 * Bootstrap particle filter. When particles, the bootstrap particle filter
 * always simulates random variables eagerly, so that no analytical
 * optimizations are performed.
 */
class BootstrapParticleFilter < ParticleFilter {
  function handle(f:Event!) -> Real {
    auto w <- 0.0;
    while f? {
      auto evt <- f!;
      if evt.isFactor() {
        w <- w + evt.observe();
      } else if evt.isRandom() {
        if evt.hasValue() {
          w <- w + evt.observe();
        } else {
          evt.simulate();
        }
      }
    }
    return w;
  }
}
