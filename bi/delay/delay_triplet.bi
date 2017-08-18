/**
 * Demonstrates sampling from a triplet of Gaussian random variables, with
 * zero or more of them given a value on the command line.
 *
 *   - `-x` : Value of the first variable.
 *   - `-y` : Value of the second variable.
 *   - `-z` : Value of the third variable.
 */
program delay_triplet(x:Gaussian, y:Gaussian, z:Gaussian) {
  delay_triplet_diagnostics();

  x ~ Gaussian(0.0, 1.0);
  y ~ Gaussian(x, 1.0);
  z ~ Gaussian(y, 1.0);
  
  stdout.printf("x = %f\n", x);
  stdout.printf("y = %f\n", y);
  stdout.printf("z = %f\n", z);
}

/*
 * Set up diagnostics.
 */
function delay_triplet_diagnostics() {
  o:DelayDiagnostics(3);
  delayDiagnostics <- o;

  o.name(1, "x");
  o.name(2, "y");
  o.name(3, "z");
}
