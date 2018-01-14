/**
 * Demonstrates how delayed sampling can yield to different runtime states
 * through a stochastic branch, inspired by a spike-and-slab prior.
 */
program delay_spike_and_slab() {
  ρ:Random<Boolean>;
  x:Random<Real>;
  y:Random<Real>;
  
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
