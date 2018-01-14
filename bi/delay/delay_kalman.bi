/**
 * Demonstrates sampling from a univariate linear-Gaussian state-space model.
 *
 *   - `-a`            : Autoregressive coefficient.
 *   - `-T`            : Number of time steps.
 */
program delay_kalman(a:Real <- 0.9, T:Integer <- 10) {
  x:Random<Real>[T];  // state
  y:Random<Real>[T];  // observation
  
  /* simulate data */
  for (t:Integer in 1..T) {
    y[t] <- simulate_gaussian(0.0, 1.0);
  }

  /* initialise */
  x[1] ~ Gaussian(0.0, 1.0);
  y[1] ~ Gaussian(x[1], 1.0);
  
  /* transition */
  for (t:Integer in 2..T) {
    x[t] ~ Gaussian(a*x[t - 1], 1.0);
    y[t] ~ Gaussian(x[t], 1.0);
  }
  
  /* output */
  for (t:Integer in 1..T) {
    stdout.print(x[t] + "\n");
  }
}
