/*
 * Test deep clone of a fiber, where a fiber is cloned, the original
 * modified, and then that original accessed via an alias pointer.
 */
program test_fiber_deep_clone_alias() {
  /* create a fiber */
  auto f <- deep_clone_fiber();
  f?;
  
  /* alias it */
  auto g <- f;
  
  /* clone it */
  auto h <- clone<Integer!>(f);

  /* modify the original */
  f?;
  
  /* check that the alias has redirected */
  if g! != 1 || h! != 1 {
    exit(1);
  }
}
