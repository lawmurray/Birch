/**
 * Sample from the posterior distribution.
 *
 *   - `-T`: number of time steps.
 *   - `-N`: number of particles.
 *   - `--ess-rel`: ESS threshold, as proportion of `N`, under which
 *     resampling is triggered.
 */
program sample(T:Integer <- 100, N:Integer <- 100, ess_rel:Real <- 0.5) {
  x:Real![N];     // particles
  w:Real[N];      // log-weights
  a:Integer[N];   // ancestor indices
  W:Real <- 0.0;  // marginal likelihood

  /* observations */
  y:Integer?[T] <- input("data/yap_dengue_trim.csv", T);

  /* initialize */
  for (n:Integer in 1..N) {
    x[n] <- particle(T, y);
    w[n] <- 0.0;
  }

  n:Integer <- 1;
  terminate:Boolean <- false;
  resample:Boolean <- false;
  while (!terminate) {    
    /* propagate and weight */
    for (n in 1..N) {
      if (x[n]?) {
        w[n] <- w[n] + x[n]!;
      } else {
        terminate <- true;
      }
    }
    resample <- ess(w) < ess_rel*N;
    //stderr.print(ess(w) + "\n");

    if (terminate || resample) {
      /* update marginal log-likelihood estimate */
      W <- W + log_sum_exp(w) - log(N);
    }
    if (resample) {
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

function input(path:String, T:Integer) -> Integer?[_] {
  y:Integer?[T];
  input:FileInputStream;
  input.open(path);
  t:Integer <- input.readInteger();
  u:Integer <- input.readInteger();
  while (!input.eof() && t <= T) {
    y[t] <- u;
    t <- input.readInteger();
    u <- input.readInteger();
  }
  input.close();
  return y;
}

/**
 * A particle.
 */
closed fiber particle(T:Integer, y:Integer?[T]) -> Real! {
  x:YapDengue(T);
  x.run(y);
  x.output();
}
