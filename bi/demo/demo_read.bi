/**
 * Shows how to use a weak reference.
 */
program demo_read() {
  a1:DemoReadA;
  a2:DemoReadA';  // read-only
  b1:DemoReadB;
  b2:DemoReadB';  // read-only

  /* OK! can assign normal references to read-only references */
  a2 <- a1;
  b2 <- a1;
  b2 <- b1;
 
  /* ERROR! cannot assign read-only references to normal references */
  //a1 <- a2;
  //b1 <- a2;
  //b1 <- b2;
}

class DemoReadA < DemoReadB {
  //
}

class DemoReadB {
  //
}