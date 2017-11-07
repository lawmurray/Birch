/**
 * Demonstrates the core ideas of delayed sampling from a minimal example
 * where pruning of the $M$-path is required before a second graft.
 *
 *   - `--diagnostics` : Enable/disable delayed sampling diagnostics.
 */
program delay_canonical(diagnostics:Boolean <- false) {
  if (diagnostics) {
    delay_canonical_diagnostics();
  }

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

/*
 * Set up diagnostics.
 */
function delay_canonical_diagnostics() {
  o:DelayDiagnostics(5);
  delayDiagnostics <- o;

  o.name(1, "a");
  o.name(2, "b");
  o.name(3, "c");
  o.name(4, "d");
  o.name(5, "e");
  
  o.position(1, 1, 1);
  o.position(2, 2, 1);
  o.position(3, 2, 2);
  o.position(4, 3, 1);
  o.position(5, 4, 1);
}
