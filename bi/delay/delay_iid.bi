/**
 * Demonstrates multiple observations in an array, used to estimate a
 * single parameter.
 *
 *   - `-μ`            : True mean of the observations.
 *   - `-σ2`           : True variance of the observations.
 *   - `-N`            : Number of observations.
 *   - `--diagnostics` : Enable/disable delayed sampling diagnostics.
 *
 * Outputs a sample from the posterior distribution of the mean, assuming a
 * `Gaussian(0.0, 1.0)` prior and Gaussian likelihood with known variance.
 */
program delay_iid(μ:Real <- 0.0, σ2:Real <- 1.0, N:Integer <- 100,
    diagnostics:Boolean <- false) {
  if (diagnostics) {
    delay_iid_diagnostics(N);
  }

  x:Gaussian;
  y:Gaussian[N];
  
  /* simulate data */
  for (n:Integer in 1..N) {
    y[n] <- simulate_gaussian(μ, σ2);
  }
  
  /* prior */
  x ~ Gaussian(0.0, 1.0);
  
  /* likelihood */
  for (n:Integer in 1..N) {
    y[n] ~ Gaussian(x, 1.0);
  }
  
  /* output */
  stdout.printf("x = %f\n", x);
}

/*
 * Set up diagnostics.
 */
function delay_iid_diagnostics(N:Integer) {
  o:DelayDiagnostics(N + 1);
  delayDiagnostics <- o;

  o.name(1, "x");
  o.position(1, (N + 1)/2, 2);

  for (n:Integer in 1..N) {
    o.name(n + 1, "y[" + n + "]");
    o.position(n + 1, n, 1);
  }
}
