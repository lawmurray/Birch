/**
 * Demonstrates sequences.
 */
program demo_sequence() {
  /* call a function with a sequence */
  sequence_function([1, 2, 3, 4, 5]);
  
  /* initialize a vector with a sequence */
  x:Real[_] <- [1, 2, 3, 4];
  
  for (i:Integer in 1..length(x)) {
    stdout.print(x[i] + " ");
  }
  stdout.print("\n");
}

function sequence_function(x:[Integer]) {
  //
}
