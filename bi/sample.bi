/**
 * Sample from the posterior distribution.
 *
 *   - `--start-time`: start time.
 *   - `--end-time`: end time.
 *   - `--nsamples`: number of samples.
 *   - `--nparticles`: number of particles to use in SMC.
 *   - `--ess-rel`: ESS threshold, as proportion of `N`, under which
 *     resampling is triggered.
 */
program sample(
    start_time:Integer <- 1,
    end_time:Integer <- 100,
    nsamples:Integer <- 1,
    nparticles:Integer <- 100,
    ess_rel:Real <- 0.7) {
  /* read observations */
  input:InputStream;
  input.open("data/yap_dengue.csv");

  y:Integer?[end_time - start_time + 1];
  
  ntimes:Integer <- 0;  // number of observations
  i:Integer <- input.readInteger();
  j:Integer <- input.readInteger();
  while (!input.eof()) {
    if (start_time <= i && i <= end_time) {
      y[i - start_time + 1] <- j;
      ntimes <- ntimes + 1;
    }
    i <- input.readInteger();
    j <- input.readInteger();
  }
  input.close();
  
  /* output times */
  out:OutputStream;
  out.open("results/yap_dengue/t.csv", "w");
  for (t:Integer in start_time..end_time) {
    out.print(" " + t);
  }
  out.print("\n");
  out.close();

  /* sample */
  x:Real![nparticles];     // particles
  w:Real[nparticles];      // log-weights
  a:Integer[nparticles];   // ancestor indices
  Z:Real;                  // marginal log-likelihood estimate
    
  for (s:Integer in 1..nsamples) {
    /* initialize */
    for (n:Integer in 1..nparticles) {
      x[n] <- particle(end_time - start_time + 1, y);
      w[n] <- 0.0;
    }
    Z <- 0.0;

    /* filter */
    for (t:Integer in 1..ntimes) {
      /* resample */
      if (t > 1 && ess(w) < ess_rel*nparticles) {
        Z <- Z + log_sum_exp(w) - log(nparticles);
        a <- ancestors(w);
        for (n:Integer in 1..nparticles) {
          if (a[n] != n) {
            x[n] <- x[a[n]];
          }
          w[n] <- 0.0;
        }
      }      
    
      /* propagate and weight */
      for (n:Integer in 1..nparticles) {
        if (x[n]?) {
          w[n] <- w[n] + x[n]!;
        } else {
          w[n] <- -inf;
        }
      }
    }
    Z <- Z + log_sum_exp(w) - log(nparticles);
    
    /* output */
    b:Integer <- ancestor(w); // returns zero if degenerate
    if (b > 0 && !x[b]?) {
      // ^ runs a chosen particle to the end, allowing it to output      
      out:OutputStream;
      out.open("results/yap_dengue/Z.csv", "a");
      out.print(Z + "\n");
      out.close();
    }
  }
}

/**
 * A particle.
 */
closed fiber particle(T:Integer, y:Integer?[_]) -> Real! {
  x:YapDengue(T);
  x.run(y);
  x.output();
}
