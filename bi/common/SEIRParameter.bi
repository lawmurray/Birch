/**
 * Parameters of an SEIR model.
 */
class SEIRParameter < Model {
  ν:Random<Real>;   // birth probability
  μ:Random<Real>;   // survival probability
  λ:Random<Real>;   // exposure probability
  δ:Random<Real>;   // infection probability
  γ:Random<Real>;   // recovery probability
  
  fiber simulate() -> Real! {
    ν <- 0.0;
    μ <- 1.0;
    λ ~ Beta(1.0, 1.0);
    δ ~ Beta(1.0, 1.0);
    γ ~ Beta(1.0, 1.0);
  }

  function output(writer:Writer) {
    writer.setReal("ν", ν);
    writer.setReal("μ", μ);
    writer.setReal("λ", λ);
    writer.setReal("δ", δ);
    writer.setReal("γ", γ);
  }
}
