/**
 * Bootstrap particle filter. When particles, the bootstrap particle filter
 * always simulates random variables eagerly, so that no analytical
 * optimizations are performed.
 */
class BootstrapParticleFilter < ParticleFilter {
  function propagate() -> Boolean {
    auto continue <- true;
    parallel for auto n in 1..nparticles {
      auto f <- x[n].step();
      auto v <- w[n];
      while f? {
        auto evt <- f!;
        if evt.isFactor() {
          v <- v + evt.observe();
        } else if evt.isRandom() {
          if evt.hasValue() {
            v <- v + evt.observe();
          } else {
            evt.simulate();
          }
        }
      }
      w[n] <- v;
    }
    return continue;
  }
}
