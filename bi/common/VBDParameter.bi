/**
 * Vector-borne disease model parameters.
 */
class VBDParameter < Model {
  h:SEIRParameter;  // humans
  m:SEIRParameter;  // mosquitos

  fiber simulate() -> Real! {
    h.ν <- 0.0;
    h.μ <- 1.0;
    h.λ ~ Beta(1.0, 1.0);
    h.δ ~ Beta(1.0, 1.0);
    h.γ ~ Beta(1.0, 1.0);
    
    m.ν <- 1.0/7.0;
    m.μ <- 6.0/7.0;
    m.λ ~ Beta(1.0, 1.0);
    m.δ ~ Beta(1.0, 1.0);
    m.γ <- 0.0;
  }

  function output(writer:Writer) {
    h.output(writer.setObject("h"));
    m.output(writer.setObject("m"));
  }
}
