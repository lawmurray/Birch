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
  
  x:YapModel(T);
  w:Real;
  f:Real! <- x.run();
  while (f?) {
    w <- w + f!;
  }
  stdout.print(w + "\n");
}

/**
 * Set up diagnostics.
 */
function configure_diagnostics(T:Integer) {
  nθ:Integer <- 11;
  nx:Integer <- 17;

  o:DelayDiagnostics(nθ + nx*T);
  
  o.name(1, "h.ν");
  o.name(2, "h.μ");
  o.name(3, "h.λ");
  o.name(4, "h.δ");
  o.name(5, "h.γ");
  o.name(6, "m.ν");
  o.name(7, "m.μ");
  o.name(8, "m.λ");
  o.name(9, "m.δ");
  o.name(10, "m.γ");
  o.name(11, "ρ");  

  for (t:Integer in 1..T) {
    offset:Integer <- nθ + nx*(t - 1);
  
    o.name(offset + 1, "h.Δs[" + t + "]");
    o.name(offset + 2, "h.Δe[" + t + "]");
    o.name(offset + 3, "h.Δi[" + t + "]");
    o.name(offset + 4, "h.Δr[" + t + "]");
    o.name(offset + 5, "h.s[" + t + "]");
    o.name(offset + 6, "h.e[" + t + "]");
    o.name(offset + 7, "h.i[" + t + "]");
    o.name(offset + 8, "h.r[" + t + "]");
    o.name(offset + 9, "m.Δs[" + t + "]");
    o.name(offset + 10, "m.Δe[" + t + "]");
    o.name(offset + 11, "m.Δi[" + t + "]");
    o.name(offset + 12, "m.Δr[" + t + "]");
    o.name(offset + 13, "m.s[" + t + "]");
    o.name(offset + 14, "m.e[" + t + "]");
    o.name(offset + 15, "m.i[" + t + "]");
    o.name(offset + 16, "m.r[" + t + "]");
    o.name(offset + 17, "y[" + t + "]");
  }
  
  delayDiagnostics <- o;
}
