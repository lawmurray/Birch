/**
 * Demonstrates how delayed sampling can yield to different runtime states
 * through a stochastic branch, inspired by a spike-and-slab prior. Outputs
 * whether the variable `y` is marginalized or realized at the end of the
 * program. This is random in each run.
 */
program delay_spike_and_slab() {  
  x:Boolean;
  y:Gaussian;
  
  x <~ Bernoulli(0.5);
  if (x) {
    y ~ Gaussian(0.0, 1.0);
  } else {
    y <- 0.0;
  }
  
  print("y is ");
  if (y.isMarginalized()) {
    print("marginalized");
  } else {
    print("realized");
  }
  print("\n");
}
