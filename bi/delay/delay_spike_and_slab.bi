/**
 * Demonstrates how delayed sampling can yield to different runtime states
 * through a stochastic branch.
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
  
  print("y is a ");
  if (y.isMissing()) {
    print("distribution");
  } else {
    print("value");
  }
  print("\n");
}
