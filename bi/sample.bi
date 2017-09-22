/**
 * Sample from the posterior distribution.
 *
 *   - `-T`            : Number of time steps.
 *   - `--diagnostics` : Enable/disable delayed sampling diagnostics.
 */
program sample(T:Integer <- 10, diagnostics:Boolean <- false) {
  if (diagnostics) {
    configure_diagnostics(T);
  }
  /*x:SEIRDiscrete(T);
  f:Real! <- x.run();
  w:Real;
  while (f?) {
    w <- f!;
  }
  x.output();*/
}

/**
 * Set up diagnostics.
 */
function configure_diagnostics(T:Integer) {
  nvars:Integer <- 5;
  o:DelayDiagnostics(nvars*T);
  delayDiagnostics <- o;
  
  for (t:Integer in 1..T) {
    o.name(nvars*(t - 1), "λ[" + t + "]");
    o.name(nvars*(t - 1), "δ[" + t + "]");
    o.name(nvars*(t - 1), "γ[" + t + "]");
    o.name(nvars*(t - 1), "y_δ[" + t + "]");
    o.name(nvars*(t - 1), "y_R[" + t + "]");
  }
}
