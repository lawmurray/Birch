/**
 * SEIR process model.
 */
class SEIRState {
  x:SEIRState?;     // previous state

  Δs:Integer;   // newly susceptible (births)
  Δe:Integer;   // newly exposed
  Δi:Integer;   // newly infected
  Δr:Integer;   // newly recovered

  s:Integer;    // susceptible population
  e:Integer;    // incubating population
  i:Integer;    // infectious population
  r:Integer;    // recovered population

  n:Integer;     // total population
  
  /**
   * Initial state.
   *
   *   - θ: parameters.
   */
  fiber run(θ:SEIRParameter) -> Real! {
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
   *   - x: previous state.
   *   - θ: parameters.
   */
  fiber run(x:SEIRState, θ:SEIRParameter) -> Real! {
    run(x, θ, Integer(ceil(Real(x.s*x.i)/x.n)), x.e, x.i);
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
  
  function output(prefix:String) {
    sout:FileOutputStream;
    eout:FileOutputStream;
    iout:FileOutputStream;
    rout:FileOutputStream;
    nout:FileOutputStream;
    
    sout.open(prefix + "s.csv", "a");
    eout.open(prefix + "e.csv", "a");
    iout.open(prefix + "i.csv", "a");
    rout.open(prefix + "r.csv", "a");
    nout.open(prefix + "n.csv", "a");

    output(sout, eout, iout, rout, nout);

    sout.print("\n");
    eout.print("\n");
    iout.print("\n");
    rout.print("\n");
    nout.print("\n");
    
    sout.close();
    eout.close();
    iout.close();
    rout.close();
    nout.close();
  }
  
  function output(sout:FileOutputStream, eout:FileOutputStream,
      iout:FileOutputStream, rout:FileOutputStream, nout:FileOutputStream) {
    if (x?) {
      x!.output(sout, eout, iout, rout, nout);
    }
    sout.print(" " + s);
    eout.print(" " + e);
    iout.print(" " + i);
    rout.print(" " + r);
    nout.print(" " + n);
  }
}
