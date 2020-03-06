/*
 * Test deep clone of an object, where clones are chained.
 */
program test_deep_clone_chain() {
  x:DeepCloneNode;
  x.a <- 1;
  x.b <- x.a;
  x.c <- 1;
  
  /* clone, modify */
  auto y <- clone(x);
  y.a <- 2;
  y.c <- 2;
  
  /* clone the clone, modify */
  auto z <- clone(y);
  z.a <- 3;
  z.c <- 3;
  
  /* erase the history of the chain */
  x <- z;
  y <- z;
  
  /* is z.b updated despite the missing history? */
  if z.b.value() != 3 {
    exit(1);
  }
}

class DeepCloneNode {
  a:Boxed<Integer>;
  b:Boxed<Integer>;
  c:Integer;
}
