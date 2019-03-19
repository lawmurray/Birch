type Token = Integer;

TOKEN_STRING:Token <- 1;
TOKEN_BOOLEAN:Token <- 2;
TOKEN_INTEGER:Token <- 3;
TOKEN_REAL:Token <- 4;
TOKEN_COLON:Token <- 5;
TOKEN_COMMA:Token <- 6;
TOKEN_LEFT_BRACE:Token <- 7;
TOKEN_RIGHT_BRACE:Token <- 8;
TOKEN_LEFT_BRACKET:Token <- 9;
TOKEN_RIGHT_BRACKET:Token <- 10;
TOKEN_VALUE:Token <- 11;

/**
 * Abstract parser. This provides basic functionality for constructing
 * objects of the Data hierarchy from input files. Subclasses inherit from
 * this and implement the abstract next() function to provide a stream of
 * tokens.
 */
class Parser {
  /**
   * Stack of values.
   */
  stack:Stack<Value>;
  
  /**
   * Current line number.
   */
  line:Integer <- 0;
  
  /**
   * Parse a file.
   */
  function parse(path:String) -> Value? {
    auto root <- stack.top();
    assert stack.empty();
    stack.pop();
    return root;
  }
  
  /**
   * Parse an object.
   */
  function object() -> Token {
    object:ObjectValue;
    token:Token;
    do {
      token <- member();
    } while token == TOKEN_COMMA;
    if token != TOKEN_RIGHT_BRACE {
      error();
    }
    return token;
  }

  /**
   * Parse an array.
   */
  function array() -> Token {
    array:ArrayValue;
    token:Token;
    do {
      token <- element();
    } while token == TOKEN_COMMA;
    if token != TOKEN_RIGHT_BRACKET {
      error();
    }
    return token;
  }
  
  /**
   * Parse a member of an object.
   */
  function member() -> Token {
    if key() != TOKEN_STRING {
      error();
    }
    if next() != TOKEN_COLON {
      error();
    }
    if value() != TOKEN_VALUE {
      error();
    }
    return next();
  }
  
  /**
   * Parse an element of an array.
   */
  function element() -> Token {
    if value() != TOKEN_VALUE {
      error();
    }
    return next();
  }
  
  function key() -> Token {
    auto token <- next();
    if token != TOKEN_STRING {
      error();
    }
    return token;
  }
  
  function value() -> Token {
    auto token <- next();
    if token == TOKEN_LEFT_BRACE {
      object();
    } else if token == TOKEN_LEFT_BRACKET {
      array();
    } else if token != TOKEN_VALUE {
      error();
    }
    return token;
  }
  
  /**
   * Next token.
   */
  function next() -> Token;
  
  /**
   * Report error.
   */
  function error() {
  
  }
}
