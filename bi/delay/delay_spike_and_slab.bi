/**
 * Demonstrates how delayed sampling can yield to different runtime states
 * through a stochastic branch, inspired by a spike-and-slab prior.
 */
program delay_spike_and_slab(diagnostics:Boolean <- false) {
  if (diagnostics) {
    delay_spike_and_slab_diagnostics();
  }

  ρ:Bernoulli;
  x:Gaussian;
  y:Gaussian;
  
  ρ ~ Bernoulli(0.5);
  x ~ Gaussian(0.0, 1.0);
  if (ρ) {
    y ~ Gaussian(x, 1.0);
    stdout.print("slab\n");
  } else {
    y ~ Gaussian(0.0, 1.0);
    stdout.print("spike\n");
  }
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

  o.position(1, 1, 1);
  o.position(2, 2, 1);
  o.position(3, 3, 1);
}
