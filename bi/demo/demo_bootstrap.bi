/**
 * Demonstrates a simple bootstrap particle filter over a linear-Gaussian
 * state-space model.
 *
 *   - `-N` : Number of particles.
 *   - `-T` : Number of time steps.
 */
program demo_bootstrap(N:Integer <- 10, T:Integer <- 10) {  
  x:Real![N];     // particles
  w:Real[N];      // log-weights
  a:Integer[N];   // ancestor indices
  W:Real <- 0.0;  // marginal likelihood

  /* initialize */
  for (n:Integer in 1..N) {
    x[n] <- run(T);
    if (x[n]?) {
      w[n] <- x[n]!;
    } else {
      w[n] <- -inf;
    }
  }
  W <- log_sum_exp(w) - log(Real(N));

  for (t:Integer in 2..T) {
    /* resample */
    a <- ancestors(w);
    for (n:Integer in 1..N) {
      if (a[n] != n) {
        x[n] <- x[a[n]];
      }
    }
    
    /* propagate and weight */
    for (n:Integer in 1..N) {
      if (x[n]?) {
        w[n] <- x[n]!;
      } else {
        w[n] <- -inf;
      }
    }
    
    /* marginal log-likelihood estimate */
    W <- W + log_sum_exp(w) - log(Real(N));
  }
    
  /* output */
  stdout.print(W + "\n");
}


class BootstrapExample(T:Integer) {
  a:Real <- 0.8;
  σ2:Real <- 1.0;
  x:Gaussian[T];
  y:Gaussian[T];

  fiber simulate() -> Real! {
    x[1] <~ Gaussian(0.0, σ2);
    y[1] ~> Gaussian(x[1], σ2);
    for (t:Integer in 2..T) {
      x[t] <~ Gaussian(a*x[t-1], σ2);
      y[t] ~> Gaussian(x[t], σ2);
    }
  }

  function input() {
    v:Real[T];
    v <- read("data/y.csv", T);    
    for (t:Integer in 1..T) {
      y[t] <- v[t];
    }
  }
}

fiber run(T:Integer) -> Real! {
  x:BootstrapExample(T);
  x.input();
  
  f:Real! <- x.simulate();
  for (t:Integer in 1..T) {
    if (f?) {
      yield f!;
    } else {
      yield -inf;
    }
  }
}
