/**
 * State model for Yap case study.
 */
class YapState < VBDState {
  z:Integer;   // actual number of new cases since last observation
  y:Integer?;  // observed number of cases since last observation

  fiber simulate(θ:YapParameter) -> Real! {
    h.n <- 7370;
    h.i <- 1 + simulate_poisson(5.0);
    h.e <- simulate_poisson(5.0);
    h.r <- Integer(simulate_uniform(0.0, h.n - h.i - h.e));
    h.s <- h.n - h.e - h.i - h.r;

    h.Δs <- 0;
    h.Δe <- h.e;
    h.Δi <- h.i;
    h.Δr <- 0;
    
    m.n <- Integer(h.n*pow(10.0, simulate_uniform(-1.0, 2.0)));
    m.r <- 0;
    m.i <- 0;
    m.e <- 0;
    m.s <- m.n;

    m.Δs <- 0;
    m.Δe <- 0;
    m.Δi <- 0;
    m.Δr <- 0;

    if (y?) {
      y! ~> Binomial(z, θ.ρ);
      z <- 0;
    } else {
      z <- h.Δi;
    }
  }
  
  fiber simulate(x:YapState, θ:YapParameter) -> Real! {
    super.simulate(x, θ);
    z <- z + x.h.Δi;
    if (y?) {
      y! ~> Binomial(z, θ.ρ);
      z <- 0;
    }
  }
  
  function input(reader:Reader) {
    y <- reader.getInteger("y");
  }
}
