/*
 * Translation of a random variable.
 */
class TranslateExpression<Value> {  
  /**
   * Random variable.
   */
  x:Random<Value>;

  /**
   * Offset.
   */
  c:Value;

  /**
   * Value conversion.
   */
  operator -> Value {
    return value();
  }
  
  /**
   * Value conversion.
   */
  function value() -> Value {
    return x.value() + c;
  }
  
  /**
   * Initialize.
   */
  function initialize(x:Random<Value>, c:Value) {
    this.x <- x;
    this.c <- c;
  }
}

operator (+x:TranslateExpression<Integer>) -> TranslateExpression<Integer> {
  return x;
}

operator (x:Random<Integer> + c:Integer) -> TranslateExpression<Integer> {
  y:TranslateExpression<Integer>;
  y.initialize(x, c);
  return y;
}

operator (x:Random<Integer> + c:Random<Integer>) -> TranslateExpression<Integer> {
  /* favour keeping the random variable on the left */
  return x + c.value();
}

operator (x:Random<Integer> + c:TranslateExpression<Integer>) -> TranslateExpression<Integer> {
  /* favour keeping the random variable on the left */
  return x + c.value();
}

operator (x:TranslateExpression<Integer> + c:Integer) -> TranslateExpression<Integer> {
  y:TranslateExpression<Integer>;
  y.initialize(x.x, x.c + c);
  return y;
}

operator (x:TranslateExpression<Integer> + c:Random<Integer>) -> TranslateExpression<Integer> {
  /* favour keeping the random variable on the left */
  return x + c.value();
}

operator (x:TranslateExpression<Integer> + c:TranslateExpression<Integer>) -> TranslateExpression<Integer> {
  /* favour keeping the random variable on the left */
  return x + c.value();
}

operator (c:Integer + x:Random<Integer>) -> TranslateExpression<Integer> {
  return x + c;
}

operator (c:Integer + x:TranslateExpression<Integer>) -> TranslateExpression<Integer> {
  return x + c;
}

operator (x:Random<Integer> - c:Integer) -> TranslateExpression<Integer> {
  y:TranslateExpression<Integer>;
  y.initialize(x, -c);
  return y;
}

operator (x:Random<Integer> - c:Random<Integer>) -> TranslateExpression<Integer> {
  /* favour keeping the random variable on the left */
  return x - c.value();
}

operator (x:Random<Integer> - c:TranslateExpression<Integer>) -> TranslateExpression<Integer> {
  /* favour keeping the random variable on the left */
  return x - c.value();
}

operator (x:TranslateExpression<Integer> - c:Integer) -> TranslateExpression<Integer> {
  y:TranslateExpression<Integer>;
  y.initialize(x.x, x.c - c);
  return y;
}

operator (x:TranslateExpression<Integer> - c:Random<Integer>) -> TranslateExpression<Integer> {
  /* favour keeping the random variable on the left */
  return x - c.value();
}

operator (x:TranslateExpression<Integer> - c:TranslateExpression<Integer>) -> TranslateExpression<Integer> {
  /* favour keeping the random variable on the left */
  return x - c.value();
}
