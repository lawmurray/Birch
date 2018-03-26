/**
 * Parameter model for Yap case study.
 */
class YapParameter < VBDParameter {
  ρ:Random<Real>;  // probability of an actual case being observed
  
  fiber parameter() -> Real {
    h.ν <- 0.0;
    h.μ <- 1.0;
    h.λ ~ Beta(1.0, 1.0);
    h.δ ~ Beta(1.0 + 2.0/4.4, 3.0 - 2.0/4.4);
    h.γ ~ Beta(1.0 + 2.0/4.5, 3.0 - 2.0/4.5);

    m.ν <- 1.0/7.0;
    m.μ <- 6.0/7.0;
    m.λ ~ Beta(1.0, 1.0);
    m.δ ~ Beta(1.0 + 2.0/6.5, 3.0 - 2.0/6.5);
    m.γ <- 0.0;

    ρ ~ Beta(1.0, 1.0);
  }
  
  function output(writer:Writer) {
    super.output(writer);
    writer.setReal("ρ", ρ);
  }  
}
