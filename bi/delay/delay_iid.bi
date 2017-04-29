/**
 * Demonstration of multiple observations in an array, used to estimate a
 * single parameter.
 *
 * `μ` True mean of the observations.
 * `σ` True standard deviation of the observations.
 * `N` Number of observations.
 *
 * Outputs a sample from the posterior distribution of the mean, assuming a
 * `Gaussian(0.0, 1.0)` prior and Gaussian likelihood with known standard
 * deviation.
 */
program delay_iid(μ:Real <- 0.0, σ:Real <- 1.0, N:Integer <- 100) {
  x:Gaussian;
  y:Gaussian[N];
  n:Integer;
  
  /* simulate data */
  n <- 0;
  while (n < N) {
    y[n] <~ Gaussian(μ, σ);
    n <- n + 1;
  }
  
  /* prior */
  x ~ Gaussian(0.0, 1.0);
  
  /* likelihood */
  n <- 0;
  while (n < N) {
    y[n] ~ Gaussian(x, 1.0);
    n <- n + 1;
  }
  
  /* output */
  print("x = ");
  print(x);
  print("\n");
}
