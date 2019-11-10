/*
 * Test deep clone of an object, where clones are chained.
 */
program test_fiber_deep_clone_chain() {
  auto f <- deep_clone_fiber();
  f?;
  
  /* clone, modify */
  auto g <- clone<Integer!>(f);
  g?;
  
  /* clone again in a chain, alias, modify */
  auto h <- clone<Integer!>(g);
  h?;
  
  /* erase the history of the chain */
  f <- h;
  g <- h;
  
  /* is h updated despite the missing history? */
  if h! != 3 {
    exit(1);
  }
}

fiber deep_clone_fiber() -> Integer {
  auto i <- 0;
  while true {
    i <- i + 1;
    yield i;
  }
}
