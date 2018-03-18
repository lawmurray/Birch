/*
 * Delta with a prior distribution that is a translation of any other
 * distribution.
 */
class TranslateAnyDelta < Random<Integer> {
  /**
   * Random variable.
   */
  x:Random<Integer>;

  /**
   * Offset.
   */
  c:Integer;

  function initialize(x:Random<Integer>, c:Integer) {
    super.initialize(x);
    this.x <- x;
    this.c <- c;
  }
  
  function doMarginalize() {
    //
  }
  
  function doCondition() {
    //
  }

  function doRealize() {
    if (isMissing()) {
      set(x.value() + c);
    } else {
      setWeight(x.observe(value() - c));
    }
  }
}

/**
 * Create delta distribution.
 */
function Delta(μ:TranslateExpression<Integer>) -> TranslateAnyDelta {
  x:TranslateAnyDelta;
  x.initialize(μ.x, μ.c);
  return x;
}
