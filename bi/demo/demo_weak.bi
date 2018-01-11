/**
 * Shows how to use a weak reference.
 */
program demo_weak() {
  a:DemoWeak;   // shared reference
  b:DemoWeak;   // shared reference
  
  c:DemoWeak&;  // weak reference
  d:DemoWeak?;  // optional
  
  /* the idiom is to assign the weak reference to an optional, then check if
   * the optional has a value */

  assert !d?;  // d has no value to start
   
  d <- c;
  assert !d?;  // c is initialized to empty, so d has no value
  d <- nil;
  
  c <- a;
  d <- c;
  assert d?;   // c was assigned a, so d has the value of a
  d <- nil;
  
  c <- a;      // a is a shared reference, and c a weak reference, to the same
               // object
  d <- c;
  assert d?;   // the object is accessible with the weak reference
  d <- nil;
               
  a <- b;      // the last shared reference to the object is lost
  d <- c;
  assert !d?;  // the object is no longer accessible with the weak reference
  d <- nil;
}

class DemoWeak {

}
