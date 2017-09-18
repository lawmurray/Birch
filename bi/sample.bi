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
  x:VBD(T);
  x.simulate();
  x.output();
}

/**
 * Set up diagnostics.
 */
function configure_diagnostics(T:Integer) {
  nvars:Integer <- 6;
  o:DelayDiagnostics(10 + nvars*T);
  delayDiagnostics <- o;

  /*o.name(1, "p_d_inc_h");
  o.name(2, "p_d_inf_h");
  o.name(3, "p_p_immune");
  o.name(4, "p_p_risk");
  o.name(5, "p_R0");
  o.name(6, "p_s_amp");
  o.name(7, "p_s_peak");
  o.name(8, "p_p_rep");
  o.name(9, "p_p_over");
  o.name(10, "initI");*/
  
  for (t:Integer in 1..T) {
    o.name(11 + nvars*(t - 1), "S[" + t + "]");
    o.name(12 + nvars*(t - 1), "E[" + t + "]");
    o.name(13 + nvars*(t - 1), "I[" + t + "]");
    o.name(14 + nvars*(t - 1), "R[" + t + "]");
    o.name(15 + nvars*(t - 1), "Z[" + t + "]");
    o.name(16 + nvars*(t - 1), "incidence[" + t + "]");

    o.position(11 + nvars*(t - 1), t, 6);
    o.position(12 + nvars*(t - 1), t, 5);
    o.position(13 + nvars*(t - 1), t, 4);
    o.position(14 + nvars*(t - 1), t, 3);
    o.position(15 + nvars*(t - 1), t, 2);
    o.position(16 + nvars*(t - 1), t, 1);
  }
}
