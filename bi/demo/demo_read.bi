/**
 * Shows how to use a weak reference.
 */
program demo_read() {
  a:DemoRead;   // shared reference
  b:DemoRead';  // read-only shared reference

  b <- a; // OK! can assign normal reference to read-only reference
  //a <- b; // ERROR! cannot assign read-only reference to normal reference
}

class DemoRead {
  //
}
