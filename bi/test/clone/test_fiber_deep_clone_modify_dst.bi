/*
 * Test deep clone of an object, where the destination object is modified
 * after the clone.
 */
program test_fiber_deep_clone_modify_dst() {
  /* create a fiber */
  auto f <- deep_clone_fiber();
  f?;
  
  /* clone it */
  auto g <- clone<Integer!>(f);

  /* modify the destination */
  g?;
  
  /* check that the source is unchanged */
  if f! != 1 {
    exit(1);
  }
}
