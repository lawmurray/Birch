/**
 * Demonstrates the use of a lambda function to delay the evaluation of an
 * expression.
 */
program demo_lambda() {
  /* declare a lambda function */
  x:@((Integer) -> Integer);
  
  /* assign a lambda function */
  x <- @(a:Integer) -> Integer {
    stdout.print("called lambda with argument: " + a + "\n");
    return 0;
  };
  
  /* call a lambda function */
  x(1);
}
