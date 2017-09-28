/**
 * Sample from the posterior distribution.
 *
 *   - `-T`: number of time steps.
 *   - `-N`: number of particles.
 *   - `--diagnostics`: enable/disable delayed sampling diagnostics.
 */
program sample(T:Integer <- 100, N:Integer <- 100, diagnostics:Boolean <- false) {
  x:Real![N];     // particles
  w:Real[N];      // log-weights
  a:Integer[N];   // ancestor indices
  W:Real <- 0.0;  // marginal likelihood

  /* initialize */
  for (n:Integer in 1..N) {
    x[n] <- particle(T);
    w[n] <- 0.0;
  }

  n:Integer <- 1;
  terminate:Boolean <- false;
  while (!terminate) {    
    /* propagate and weight */
    for (n in 1..N) {
      if (x[n]?) {
        w[n] <- x[n]!;
      } else {
        terminate <- true;
      }
    }
    
    if (!terminate) {
      /* marginal log-likelihood estimate */
      W <- W + log_sum_exp(w) - log(N);
 
      /* resample */
      a <- ancestors(w);
      for (n in 1..N) {
        if (a[n] != n) {
          x[n] <- x[a[n]];
        }
        w[n] <- 0.0;
      }
    }
  }
  
  stdout.print(W + "\n");
}

/**
 * A particle.
 */
closed fiber particle(T:Integer) -> Real! {
  x:YapModel(T);
  x.run();
}
