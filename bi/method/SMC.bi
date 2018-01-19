/**
 * Sequential Monte Carlo (SMC).
 */
class SMC {
  /**
   * Simulate.
   */
  function simulate(model:String, input:Reader, output:Writer, T:Integer,
      N:Integer, trigger:Real) {
    f:(Model', Real)![N];  // particles
    w:Real[N];             // log-weights
    a:Integer[N];          // ancestors
    Z:Real <- 0.0;         // marginal log-likelihood estimate
    x:Model';
    n:Integer;
    t:Integer;
    v:Real;
  
    /* initialize */
    for (n in 1..N) {
      f[n] <- particle(model, input);
      w[n] <- 0.0;
      a[n] <- n;
    }
  
    /* filter */
    for (t in 1..T) {
      if (ess(w) < trigger*N) {
        Z <- Z + log_sum_exp(w) - log(N);
        a <- ancestors(w);
        for (n in 1..N) {
          if (a[n] != n) {
            f[n] <- f[a[n]];
          }
          w[n] <- 0.0;
        }
      }
    
      /* propagate and weight */
      for (n in 1..N) {
        if (f[n]?) {
          (x, v) <- f[n]!;
          w[n] <- w[n] + v;
        } 
      }
    }
    Z <- Z + log_sum_exp(w) - log(N);
    
    /* output */
    (x, v) <- f[ancestor(w)]!;
    x.output(output);
  }
}

closed fiber particle(model:String, input:Reader') -> (Model', Real)! {
  /* create model */
  x:Model? <- Model?(make(model));
  if (!x?) {
    stderr.print("error: " + model + " must be a subtype of Model with no initialization parameters.\n");
    assert false;
  }
  
  /* input */
  x!.input(input);
  
  /* simulate */
  f:Real! <- x!.simulate();
  while (f?) {
    yield (x!, f!);
  }
}
