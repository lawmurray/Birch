/**
 * Demonstrates multiple observations in an array, used to estimate a
 * single parameter.
 *
 *   - `-μ`            : True mean of the observations.
 *   - `-σ2`           : True variance of the observations.
 *   - `-N`            : Number of observations.
 *
 * Outputs a sample from the posterior distribution of the mean, assuming a
 * `Gaussian(0.0, 1.0)` prior and Gaussian likelihood with known variance.
 */
program delay_iid(μ:Real <- 0.0, σ2:Real <- 1.0, N:Integer <- 100) {
  x:Random<Real>;
  y:Random<Real>[N];
  
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
  stdout.print("x = " + x + "\n");
}
