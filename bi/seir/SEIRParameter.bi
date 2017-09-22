/**
 * Parameters of an SEIR model.
 */
class SEIRParameter {
  ν:Gamma;  // birth rate
  μ:Beta;   // mortality probability
  λ:Beta;   // exposure probability
  δ:Beta;   // infection probability
  γ:Beta;   // recovery probability
  
  fiber run() -> Real! {
    ν <- 0.0;
    μ <- 0.0;
    λ ~ Beta(1.0, 1.0);
    δ ~ Beta(1.0, 1.0);
    γ ~ Beta(1.0, 1.0);
  }
}
