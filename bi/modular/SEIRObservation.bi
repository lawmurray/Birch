/**
 * SEIR obervation model.
 */
class SEIRObservation {
  ρ:Beta;      // reporting probability
  y:Binomial;  // observed number of newly infected

  function parameter() {
    ρ ~ Beta(1.0, 1.0);
  }
  
  function observe(x:SEIRState) {
    y ~ Binomial(x.Δi, ρ);
  }
}
