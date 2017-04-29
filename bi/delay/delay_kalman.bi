/**
 * Demonstrates sampling from a univariate linear-Gaussian state-space model.
 *
 * `a` Autoregressive coefficient.
 * `T` Number of time steps.
 */
program delay_kalman(a:Real <- 0.9, T:Integer <- 10) {
  x:Gaussian[T];  // state
  y:Gaussian[T];  // observation
  t:Integer;
  
  /* simulate data */
  t <- 0;
  while (t < T) {
    y[t] <~ Gaussian(0.0, 1.0);
    t <- t + 1;
  }

  /* initialise */
  x[0] ~ Gaussian(0.0, 1.0);
  y[0] ~ Gaussian(x[0], 1.0);
  
  /* transition */
  t <- 1;
  while (t < T) {
    x[t] ~ Gaussian(a*x[t - 1], 1.0);
    y[t] ~ Gaussian(x[t], 1.0);
    t <- t + 1;
  }
  
  /* output */
  t <- 0;
  while (t < T) {
    print(x[t]);
    print("\n");
    t <- t + 1;
  }
}
