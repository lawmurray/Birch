/**
 * Demonstrates the core ideas of delayed sampling from a minimal example
 * where pruning of the $M$-path is required before a second graft.
 */
program delay_canonical() {
  a:Random<Real>;
  b:Random<Real>;
  c:Random<Real>;
  d:Random<Real>;
  e:Random<Real>;
  
  e <- 4.0;

  a ~ Gaussian(0.0, 1.0);
  b ~ Gaussian(a, 1.0);
  c ~ Gaussian(b, 1.0);
  d ~ Gaussian(b, 1.0);
  e ~ Gaussian(d, 1.0);
  
  stdout.print(c + "\n");
}
