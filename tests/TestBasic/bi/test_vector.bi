/*
 * Test Vector.
 */
program test_vector() {
  o:Vector<Integer>;
  
  o.pushBack(1);
  o.pushBack(2);
  o.pushBack(4);
  o.pushBack(5);
  if !check_vector(o, [1, 2, 4, 5]) {
    exit(1);
  }
  
  o.insert(3, 3);
  if !check_vector(o, [1, 2, 3, 4, 5]) {
    exit(1);
  }

  o.erase(4);
  if !check_vector(o, [1, 2, 3, 5]) {
    exit(1);
  }
  
  if o.front() != 1 {
    exit(1);
  }
  
  if o.back() != 5 {
    exit(1);
  }
}

function check_vector(o:Vector<Integer>, values:Integer[_]) -> Boolean {
  auto result <- true;
  
  /* number of rows */
  if o.size() != length(values) {
    stderr.print("incorrect total size\n");
    result <- false;
  }
  
  /* contents */
  for i in 1..o.size() {
    if o.get(i) != values[i] {
      stderr.print("incorrect value\n");
      result <- false;
    }
  }

  return result;
}
