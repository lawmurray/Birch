/**
 *
 */
class SEIRModel < MarkovModel<SEIRParameter,SEIRState> {
  fiber parameter(θ:SEIRParameter) -> Real {
    θ.ν <- 0.0;
    θ.μ <- 1.0;
    θ.λ ~ Beta(1.0, 1.0);
    θ.δ ~ Beta(1.0, 1.0);
    θ.γ ~ Beta(1.0, 1.0);
  }

  fiber initial(x:SEIRState, θ:SEIRParameter) -> Real {
    x.Δs <- 0;
    x.Δe <- 0;
    x.Δi <- 1;
    x.Δr <- 0;
    
    x.s <- 0;
    x.e <- 0;
    x.i <- 1;
    x.r <- 0;
    
    x.n <- 0;
  }
  
  fiber transition(x':SEIRState, x:SEIRState, θ:SEIRParameter) -> Real {
    transition(x', x, θ, (x.s*x.i + x.n - 1)/x.n, x.e, x.i);
  }
  
  fiber transition(x':SEIRState, x:SEIRState, θ:SEIRParameter, ns:Integer,
      ne:Integer, ni:Integer) -> Real {
    /* transfers */
    x'.Δe <~ Binomial(ns, θ.λ);
    x'.Δi <~ Binomial(ne, θ.δ);
    x'.Δr <~ Binomial(ni, θ.γ);

    x'.s <- x.s - x'.Δe;
    x'.e <- x.e + x'.Δe - x'.Δi;
    x'.i <- x.i + x'.Δi - x'.Δr;
    x'.r <- x.r + x'.Δr;
    
    /* deaths */
    x'.s <~ Binomial(x'.s, θ.μ);
    x'.e <~ Binomial(x'.e, θ.μ);
    x'.i <~ Binomial(x'.i, θ.μ);
    x'.r <~ Binomial(x'.r, θ.μ);

    /* births */
    x'.Δs <~ Binomial(x.n, θ.ν);
    x'.s <- x'.s + x'.Δs;
    
    /* update population */
    x'.n <- x'.s + x'.e + x'.i + x'.r;
  }
}
