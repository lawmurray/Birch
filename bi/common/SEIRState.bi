/**
 * SEIR process model.
 */
class SEIRState < State {
  Δs:Integer;  // newly susceptible (births)
  Δe:Integer;  // newly exposed
  Δi:Integer;  // newly infected
  Δr:Integer;  // newly recovered

  s:Integer;   // susceptible population
  e:Integer;   // incubating population
  i:Integer;   // infectious population
  r:Integer;   // recovered population

  n:Integer;   // total population
  
  /**
   * Initial state.
   *
   * - θ: Parameters.
   */
  fiber simulate(θ:SEIRParameter) -> Real! {
    Δs <- 0;
    Δe <- 0;
    Δi <- 1;
    Δr <- 0;
    
    s <- 0;
    e <- 0;
    i <- 1;
    r <- 0;
    
    n <- 0;
  }
  
  /**
   * Next state, with default trial counts.
   *
   * - x: Previous state.
   * - θ: Parameters.
   */
  fiber simulate(x:SEIRState, θ:SEIRParameter) -> Real! {
    simulate(x, θ, (x.s*x.i + x.n - 1)/x.n, x.e, x.i);
  }
  
  /**
   * Next state, with given trial counts.
   *
   * - x: Previous state.
   * - θ: Parameters.
   * - ns: Number of trials in susceptible population.
   * - ne: Number of trials in exposed population.
   * - ni: Number of trials in infected population.
   */
  fiber simulate(x:SEIRState, θ:SEIRParameter, ns:Integer, ne:Integer,
      ni:Integer) -> Real! {
    /* transfers */
    Δe <~ Binomial(ns, θ.λ);
    Δi <~ Binomial(ne, θ.δ);
    Δr <~ Binomial(ni, θ.γ);

    s <- x.s - Δe;
    e <- x.e + Δe - Δi;
    i <- x.i + Δi - Δr;
    r <- x.r + Δr;
    
    /* deaths */
    s <~ Binomial(s, θ.μ);
    e <~ Binomial(e, θ.μ);
    i <~ Binomial(i, θ.μ);
    r <~ Binomial(r, θ.μ);

    /* births */
    Δs <~ Binomial(x.n, θ.ν);
    s <- s + Δs;
    
    /* update population */
    n <- s + e + i + r;
  }
  
  function output(writer:Writer) {
    writer.setInteger("n", n);
    writer.setInteger("s", s);
    writer.setInteger("e", e);
    writer.setInteger("i", i);
    writer.setInteger("Δs", Δs);
    writer.setInteger("Δe", Δe);
    writer.setInteger("Δi", Δi);
    writer.setInteger("Δr", Δr);
  }
}
