/**
 * Demonstrates the use of a lambda function to delay the evaluation of an
 * expression.
 */
program demo_lambda() {
  /* declare a lambda function */
  x:lambda() -> Boolean;
  
  /* assign a lambda function */
  x <- lambda() -> y:Boolean {
    print("called lambda\n");
    y <- true;
  };
  
  /* call a lambda function */
  x();
}
