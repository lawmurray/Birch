/**
 * Demonstrates sampling from a univariate linear-Gaussian state-space model.
 *
 * `a` Autoregressive coefficient.
 * `T` Number of time steps.
 */
program delay_kalman(a:Real <- 0.9, T:Integer <- 10) {
  x:Gaussian[T];  // state
  y:Real[T];  // observation
  t:Integer;
  
  /* simulate data */
  for (t in 1..T) {
    y[t] <~ Gaussian(0.0, 1.0);
  }

  /* initialise */
  x[1] ~ Gaussian(0.0, 1.0);
  y[1] ~> Gaussian(x[1], 1.0);
  
  /* transition */
  for (t in 2..T) {
    x[t] ~ Gaussian(a*x[t - 1], 1.0);
    y[t] ~> Gaussian(x[t], 1.0);
  }
  
  /* output */
  for (t in 1..T) {
    print(x[t]);
    print("\n");
  }
}
