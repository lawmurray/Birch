class SMC {
  /**
   * Number of particles.
   */
  N:Integer <- 100;

  /**
   * Relative ESS to trigger resampling.
   */
  trigger:Real <- 0.7;

  /**
   * 
   */
  function step() -> Real {
    f:(Model', Real)![N];  // particles
    x:Model'[N];           // state
    w:Real[N];             // log-weights
    a:Integer[N];          // ancestors
    Z:Real <- 0.0;         // marginal log-likelihood
  
    /* initialize */
    for (n:Integer in 1..N) {
      f[n] <- particle();
    }
    Z <- 0.0;
  
    while (true) {
      if (ess(w) < trigger*N) {
        Z <- Z + log_sum_exp(w) - log(N);
        a <- ancestors(w);
        for (n:Integer in 1..N) {
          if (a[n] != n) {
            x[n] <- x[a[n]];
          }
          w[n] <- 0.0;
        }
      }
    
      /* propagate and weight */
      for (n:Integer in 1..N) {
        if (f[n]?) {
          (x[n], w[n]) <- f[n]!;
          // ^ add previous weight
        } 
      }
    }
    Z <- Z + log_sum_exp(w) - log(N);
  }
}

closed fiber particle() -> (Model', Real)! {
  x:Model;
  f:Real! <- x.simulate();
  while (f?) {
    yield (x, f!);
  }
}
