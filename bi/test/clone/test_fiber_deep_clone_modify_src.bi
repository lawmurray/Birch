/*
 * Test deep clone of an object, where the source object is modified
 * after the clone.
 */
program test_fiber_deep_clone_modify_src() {
  /* create a fiber */
  auto f <- deep_clone_fiber();
  f?;
  
  /* clone it */
  auto g <- clone<Integer!>(f);

  /* modify the source */
  f?;
  
  /* check that the destination is unchanged */
  if g! != 1 {
    exit(1);
  }
}
