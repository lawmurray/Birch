/**
 * Demonstrates tuples.
 */
program demo_tuple() {
  x:(Integer, Integer);
  a:Integer;
  b:Integer;
  
  /* pack */
  x <- (1, 2);
  
  /* unpack */
  (a, b) <- x;
  stdout.print("a = " + a + "\n");
  stdout.print("b = " + b + "\n");
  
  /* call a function */
  x <- tuple_function((3, 4));
  (a, b) <- tuple_function(x);
  stdout.print("a = " + a + "\n");
  stdout.print("b = " + b + "\n");
}

function tuple_function(x:(Integer, Integer)) -> (Integer, Integer) {
  return x;
}
