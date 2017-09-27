/**
 * State for Yap case study.
 */
class YapState < VBDState {
  z:Integer;   // actual number of new cases since last observation
  y:Integer?;  // observed number of new cases since last observation

  /**
   * Initial state.
   */
  fiber run(θ:YapParameter) -> Real! {
    h.n <- 100000;
    h.r <~ Binomial(h.n, 0.06);
    h.i <~ Poisson(10.0);
    h.e <- 0;
    h.s <- h.n - h.e - h.i - h.r;
    
    m.n <- Integer(h.n*pow(10.0, simulate_uniform(-1.0, 2.0)));
    m.r <- 0;
    m.i <- 0;
    m.e <- 0;
    m.s <- m.n;
    
    z <- 0;
  }
  
  /**
   * Next state.
   */
  fiber run(x:YapState, θ:YapParameter) -> Real! {
    super.run(x, θ);
    
    /* observations are of the aggregated number of new cases of infection
     * since the time of the last observation */
    z <- z + x.h.Δi;
    if (y?) {
      y! ~> Binomial(z, θ.ρ);
      z <- 0;
    }
  }
}
