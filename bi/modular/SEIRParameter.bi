/**
 * Parameters of an SEIR model.
 */
class SEIRParameter {
  ν:Gamma;  // birth rate
  μ:Beta;   // mortality probability
  λ:Beta;   // exposure probability
  δ:Beta;   // infection probability
  γ:Beta;   // recovery probability
}
