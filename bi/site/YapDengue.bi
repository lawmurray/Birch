/**
 * Model for Yap case study.
 *
 *   - T: number of time steps.
 */
class YapDengue(T:Integer) {
  θ:VBDParameter;
  x:VBDState;
  
  /* observations are of the aggregated number of new cases of infection
   * since the time of the last observation */
  z:Integer;      // actual number of new cases since last observation
  y:Integer?[T];  // observed number of new cases since last observation
  ρ:Beta;         // probability of an actual case being observed
  
  /**
   * Run the model.
   */
  fiber run(y:Integer?[_]) -> Real! {
    /* parameter model */
    θ.run();
    ρ ~ Beta(1.0, 1.0);
    
    /* initial model */
    h:SEIRState <- x.h;
    m:SEIRState <- x.m;
    
    h.n <- 7370;
    h.r <~ Binomial(h.n, 0.06);
    h.i <- 1 + simulate_poisson(10.0);
    h.e <- simulate_poisson(10.0);
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
    
    z <- h.Δi;
    if (y[1]?) {
      y[1]! ~> Binomial(z, ρ);
      z <- 0;
    }

    /* transition model */
    for (t:Integer in 2..T) {
      new:VBDState;
      new.run(x, θ);
      x <- new;
      
      z <- z + x.h.Δi;
      if (y[t]?) {
        y[t]! ~> Binomial(z, ρ);
        z <- 0;
      }      
    }
  }
    
  function output() {
    prefix:String <- "results/yap_dengue/";
    θ.output(prefix);
    x.output(prefix);
    
    output:FileOutputStream;
    output.open(prefix + "ρ.csv", "a");
    output.print(ρ + "\n");
    output.close();
  }
}
