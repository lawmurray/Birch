/**
 * Demonstrates sampling from a univariate linear-Gaussian state-space model.
 *
 *   - `-a`            : Autoregressive coefficient.
 *   - `-T`            : Number of time steps.
 *   - `--diagnostics` : Enable/disable delayed sampling diagnostics.
 */
program delay_kalman(a:Real <- 0.9, T:Integer <- 10,
    diagnostics:Boolean <- false) {
  if (diagnostics) {
    delay_kalman_diagnostics(T);
  }

  x:Gaussian[T];  // state
  y:Gaussian[T];  // observation
  
  /* simulate data */
  for (t:Integer in 1..T) {
    y[t] <- random_gaussian(0.0, 1.0);
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
    stdout.printf("%f\n", x[t]);
  }
}

/*
 * Set up diagnostics.
 */
function delay_kalman_diagnostics(T:Integer) {
  o:DelayDiagnostics(2*T);
  delayDiagnostics <- o;

  for (t:Integer in 1..T) {
    o.name(2*t - 1, "x[" + t + "]");
    o.name(2*t, "y[" + t + "]");
    
    o.position(2*t - 1, t, 2);
    o.position(2*t, t, 1);
  }
}
