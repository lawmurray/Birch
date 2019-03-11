/**
 * Abstract parser. This provides basic functionality for constructing
 * objects of the Data hierarchy from input files. Subclasses inherit from
 * this and implement the abstract next() function to provide a stream of
 * tokens.
 */
class Parser {
  /**
   * Root value.
   */
  root:Value;

  /**
   * Last value.
   */
  value:Value;
  
  /**
   * Current line number.
   */
  line:Integer <- 0;
  
  /**
   * Next token.
   */
  function next() -> Integer;
  
  /**
   * Report error.
   */
  function error() {
  
  }
}
