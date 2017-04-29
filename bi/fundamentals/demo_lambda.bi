/**
 * Demonstrates the use of a lambda function to delay the evaluation of an
 * expression.
 */
program demo_lambda() {
  /* declare a lambda function */
  x:lambda();
  
  /* assign a lambda function */
  x <- lambda() {
    print("called lambda\n");
  };
  
  /* call a lambda function */
  x();
}
