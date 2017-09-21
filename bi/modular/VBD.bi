/**
 * Vector-bourne disease model.
 */
class VBDBinomial {
  /* humans */
  θ:SEIRParameter;
  h:SEIRState;
  
  /* mosquitos */
  φ:SEIRParameter;
  m:SEIRState;
  
  /* observations */
  y:SEIRObservation;

  function parameter() {
    θ.ν <- 0.0;
    θ.μ <- 0.0;
    θ.λ ~ Beta(1.0, 1.0);
    θ.δ ~ Beta(1.0/4.4, 1.0 - 1.0/4.4);
    θ.γ ~ Beta(1.0/4.5, 1.0 - 1.0/4.5);
    
    φ.ν <- 1.0;
    φ.μ <- 1.0/7.0;  // expected lifespan of mosquito is one week
    φ.λ ~ Beta(1.0, 1.0);
    φ.δ ~ Beta(1.0/6.5, 1.0 - 1.0/6.5);
    φ.γ <- 0.0;
  }

  function initial() {
    h.n <- 100000;
    h.r <~ Binomial(n, 0.06);
    h.i <~ Poisson(10.0);
    h.e <- 0;
    h.s <- h.n - h.e - h.i - h.r;
    
    m.n <- Integer(Real(h.n)*pow(10.0, random_uniform(-1.0, 2.0)));
    m.r <- 0;
    m.i <- 0;
    m.e <- 0;
    m.s <- m.n;
  }

  function transition() {
    n:SEIRExchange;
  
    n.e <- h.s*m.i/m.n;
    n.i <- h.e;
    n.r <- h.i;
    h <- h.transition(n, θ);
    
    n.e <- m.s*h.i/h.n;
    n.i <- m.e;
    n.r <- m.i;
    m <- m.transition(n, φ);
  }
  
  function observe() {
    y.observe(h);
  }
}
