/**
 * State for Yap case study.
 */
class YapState < VBDState {
  y:Binomial;  // observable number of newly infected humans

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
  }
  
  /**
   * Next state.
   */
  fiber run(x:YapState, θ:YapParameter) -> Real! {
    super.run(x, θ);
    y ~ Binomial(x.h.Δi, θ.ρ);
  }
}
