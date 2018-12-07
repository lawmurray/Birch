/*
 * Test deep clone of an object, where clones are chained.
 */
program test_deep_clone_chain() {
  x:DeepCloneNode;
  x.a <- 1;
  x.b <- x.a;
  
  /* clone, modify */
  auto y <- clone<DeepCloneNode>(x);
  y.a <- 2;
  
  /* clone again in a chain, alias, modify */
  auto z <- clone<DeepCloneNode>(y);
  z.a <- 3;
  
  /* erase the history of the chain */
  x <- z;
  y <- z;
  
  /* is z.b updated despite the missing history? */
  if (z.b != 3) {
    exit(1);
  }
}

class DeepCloneNode {
  a:Boxed<Integer>(0);
  b:Boxed<Integer>(0);
}
