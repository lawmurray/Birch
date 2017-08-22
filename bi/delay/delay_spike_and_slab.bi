/**
 * Demonstrates how delayed sampling can yield to different runtime states
 * through a stochastic branch, inspired by a spike-and-slab prior. Outputs
 * whether the variable `y` is marginalized or realized at the end of the
 * program. This is random in each run.
 */
program delay_spike_and_slab() {
  delay_spike_and_slab_diagnostics();

  ρ:Bernoulli;
  x:Gaussian;
  y:Gaussian;
  
  ρ ~ Bernoulli(0.5);
  if (ρ) {
    x ~ Gaussian(0.0, 1.0);
  } else {
    x <- 0.0;
  }
  y ~ Gaussian(x, 1.0);
  
  stdout.print("x is ");
  if (x.isMarginalized()) {
    stdout.print("marginalized");
  } else {
    stdout.print("realized");
  }
  stdout.print("\n");
}

/*
 * Set up diagnostics.
 */
function delay_spike_and_slab_diagnostics() {
  o:DelayDiagnostics(3);
  delayDiagnostics <- o;
  
  o.name(1, "ρ");
  o.name(2, "x");
  o.name(3, "y");

  o.position(1, 1, 2);
  o.position(2, 2, 2);
  o.position(3, 2, 1);
}
