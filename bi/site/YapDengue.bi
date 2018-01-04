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
    θ.h.ν <- 0.0;
    θ.h.μ <- 1.0;
    θ.h.λ ~ Beta(1.0, 1.0);
    θ.h.δ ~ Beta(1.0 + 2.0/4.4, 3.0 - 2.0/4.4);
    θ.h.γ ~ Beta(1.0 + 2.0/4.5, 3.0 - 2.0/4.5);

    θ.m.ν <- 1.0/7.0;
    θ.m.μ <- 6.0/7.0;
    θ.m.λ ~ Beta(1.0, 1.0);
    θ.m.δ ~ Beta(1.0 + 2.0/6.5, 3.0 - 2.0/6.5);
    θ.m.γ <- 0.0;

    ρ ~ Beta(1.0, 1.0);
    
    /* initial model */
    h:SEIRState <- x.h;
    m:SEIRState <- x.m;
    
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
    prefix:String <- "output/yap_dengue/";
    
    θ.output(prefix);
    x.output(prefix);
    
    output:OutputStream;
    output.open(prefix + "ρ.csv", "a");
    output.print(ρ + "\n");
    output.close();
  }
}
