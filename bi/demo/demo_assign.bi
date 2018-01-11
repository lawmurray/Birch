/**
 * Demonstrates valid assignments.
 */
program demo_assign() {
  a:AssignA;
  b:AssignB;
  x:Real;
  
  a <- b;    // OK! AssignB is a subtype of AssignA
  //b <- a;  // ERROR! AssignA is not a subtype of AssignB

  a <- x;    // OK! AssignA declares a custom assignment operator for Real
  b <- x;    // OK! AssignB is a subtype of AssignA, and inherits the custom
             //     assignment operator
}

class AssignA {
  operator <- x:Real {
    //
  }
}

class AssignB < AssignA {
  //
}
