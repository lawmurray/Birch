/**
 * SEIR process model.
 */
class SEIRState {
  x:SEIRState?;     // previous state

  Δs:GammaPoisson;   // newly susceptible (births)
  Δe:BetaBinomial;   // newly exposed
  Δi:BetaBinomial;   // newly infected
  Δr:BetaBinomial;   // newly recovered

  s:Binomial;    // susceptible population
  e:Binomial;    // incubating population
  i:Binomial;    // infectious population
  r:Binomial;    // recovered population

  n:Integer;     // total population
  
  /**
   * Initial state.
   *
   *   - θ: parameters.
   */
  fiber run(θ:SEIRParameter) -> Real! {
    Δs <- 0;
    Δe <- 0;
    Δi <- 0;
    Δr <- 0;
    
    s <- 0;
    e <- 0;
    i <- 0;
    r <- 0;
    
    n <- 0;
  }
  
  /**
   * Next state, with default trial counts.
   *
   *   - x: previous state.
   *   - θ: parameters.
   */
  fiber run(x:SEIRState, θ:SEIRParameter) -> Real! {
    run(x, θ, x.s*x.i/x.n, x.e, x.i);
  }
  
  /**
   * Next state, with externally computed trial counts.
   *
   *   - x: previous state.
   *   - θ: parameters.
   *   - ns: number of trials in susceptible population.
   *   - ne: number of trials in exposed population.
   *   - ni: number of trials in infected population.
   */
  fiber run(x:SEIRState, θ:SEIRParameter, ns:Integer, ne:Integer,
      ni:Integer) -> Real! {
    this.x <- x;

    Δs ~ Poisson(θ.ν);
    Δe ~ Binomial(ns, θ.λ);
    Δi ~ Binomial(ne, θ.δ);
    Δr ~ Binomial(ni, θ.γ);

    s ~ Binomial(x.s + Δs - Δe, 1.0 - θ.μ);
    e ~ Binomial(x.e + Δe - Δi, 1.0 - θ.μ);
    i ~ Binomial(x.i + Δi - Δr, 1.0 - θ.μ);
    r ~ Binomial(x.r + Δr, 1.0 - θ.μ);
    
    n <- s + e + i + r;
  }
}
