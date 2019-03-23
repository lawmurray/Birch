class Particle(m:ForwardModel) {
  /**
   * The model.
   */
  auto m <- m;
  
  /**
   * Start the particle.
   */
  function start() -> Real {
    auto f <- m.start();
    auto w <- 0.0;
    while f? {
      auto evt <- f!;
      if evt.isFactor() || (evt.isRandom() && evt.hasValue()) {
        w <- w + evt.observe();
      }
    }
    return w;
  }
  
  /**
   * Step the particle.
   */
  function step() -> Real {
    auto f <- m.step();
    auto w <- 0.0;
    while f? {
      auto evt <- f!;
      if evt.isFactor() || (evt.isRandom() && evt.hasValue()) {
        w <- w + evt.observe();
      }
    }
    return w;
  }
}
