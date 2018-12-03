/*
 * Test deep clone of an object, where the destination object is modified
 * after the clone.
 */
program test_deep_clone_modify_dst() {
  /* create a simple list */
  x:List<Integer>;
  x.pushBack(1);
  x.pushBack(2);
  
  /* clone the list */
  auto y <- clone<List<Integer>>(x);
  
  /* modify the clone */
  y.set(1, 3);
  y.set(2, 4);
  
  /* check that the original is unchanged */
  if (x.get(1) != 1 || x.get(2) != 2) {
    exit(1);
  }
}
