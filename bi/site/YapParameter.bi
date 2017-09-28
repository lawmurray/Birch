/**
 * Parameters for Yap case study.
 */
class YapParameter < VBDParameter {
  ρ:Beta;  // reporting probability
  
  fiber run() -> Real! {
    h.ν <- 0.0;
    h.μ <- 0.0;
    h.λ ~ Beta(1.0, 1.0);
    h.δ ~ Beta(1.0/4.4, 1.0 - 1.0/4.4);
    h.γ ~ Beta(1.0/4.5, 1.0 - 1.0/4.5);
    
    m.ν <- 1.0/7.0;
    m.μ <- 1.0/7.0;  // expected lifespan of mosquito is one week
    m.λ ~ Beta(1.0, 1.0);
    m.δ ~ Beta(1.0/6.5, 1.0 - 1.0/6.5);
    m.γ <- 0.0;

    ρ ~ Beta(1.0, 1.0);
  }
}
