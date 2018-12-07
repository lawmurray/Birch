/*
 * Test for memory leaks when deep cloning objects.
 */
program test_deep_clone_leak() {
  test_deep_clone_leak_aux();
  refCount:Integer;
  cpp{{
  refCount_ = bi::cloneMemo->numShared();
  }}
  if (refCount != 1) {
    exit(1);
  }
}

function test_deep_clone_leak_aux() {
  /* create a simple list */
  x:List<Integer>;
  x.pushBack(1);
  x.pushBack(2);
  
  /* clone the list */
  auto y <- clone<List<Integer>>(x);

  /* modify the clone */
  y.set(1, 3);
  y.set(2, 4);

  if (x.get(1) != 1 || x.get(2) != 2) {
    //
  }
}
