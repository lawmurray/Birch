/**
 * Demonstrates valid assignments.
 */
program demo_assign() {
  a:AssignA;
  b:AssignB;
  c:Real;
  
  a <- b;
  b <- c;
}

class AssignA {
  operator <- x:Real {

  }
}

class AssignB < AssignA {

}
