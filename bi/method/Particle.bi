class Particle(m:Model) {
  /**
   * The model.
   */
  auto m <- m;
  
  /**
   * The current checkpoint of simulation.
   */
  auto checkpoint <- m.simulate();
  
  /**
   * Step the particle.
   */
  function step() -> Real {

  }
}
