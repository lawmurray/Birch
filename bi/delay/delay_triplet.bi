/**
 * Demonstrates sampling from a triplet of Gaussian random variables, with
 * zero or more of them given a value on the command line.
 *
 *   - `-x`            : Value of the first variable.
 *   - `-y`            : Value of the second variable.
 *   - `-z`            : Value of the third variable.
 *   - `--diagnostics` : Enable/disable delayed sampling diagnostics.
 */
program delay_triplet(x:Real?, y:Real?, z:Real?,
    diagnostics:Boolean <- false) {
  if (diagnostics) {
    delay_triplet_diagnostics();
  }

  x1:Random<Real>;
  y1:Random<Real>;
  z1:Random<Real>;
  
  if (x?) {
    x1 <- x!;
  }
  if (y?) {
    y1 <- y!;
  }
  if (z?) {
    z1 <- z!;
  }
  
  x1 ~ Gaussian(0.0, 1.0);
  y1 ~ Gaussian(x1, 1.0);
  z1 ~ Gaussian(y1, 1.0);
  
  stdout.print("x = " + x1 + "\n");
  stdout.print("y = " + y1 + "\n");
  stdout.print("z = " + z1 + "\n");
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
  
  o.position(1, 1, 1);
  o.position(2, 2, 1);
  o.position(3, 3, 1);
}
