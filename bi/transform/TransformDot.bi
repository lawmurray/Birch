/*
 * Linear transformation of a multivariate random variate, as represented by
 * its associated distribution, which involves a dot product.
 *
 * - Value: Distribution type.
 */
class TransformDot<Value>(a:Expression<Real[_]>, x:Value,
    c:Expression<Real>) {
  /**
   * Scale.
   */
  a:Expression<Real[_]> <- a;
  
  /**
   * Distribution.
   */
  x:Value <- x;

  /**
   * Offset.
   */
  c:Expression<Real> <- c;
   
  function multiply(y:Expression<Real>) {
    a <- a*y;
    c <- c*y;
  }

  function divide(y:Expression<Real>) {
    a <- a/y;
    c <- c/y;
  }

  function add(y:Expression<Real>) {
    c <- c + y;
  }

  function subtract(y:Expression<Real>) {
    c <- c - y;
  }
  
  function negateAndAdd(y:Expression<Real>) {
    a <- -a;
    c <- y - c;
  }
  
  /**
   * Is the transformation valid? This evaluates the scale and offset. It 
   * then returns true if the Distribution object remains uninstantiated, and
   * false otherwise (which would mean that either or both of the scale and
   * offset depend it).
   */
  function isValid() -> Boolean {
    a.value();
    c.value();
    return !x.hasValue();
  }
}

function TransformDot<Value>(a:Expression<Real[_]>, x:Value,
    c:Expression<Real>) -> TransformDot<Value> {
  m:TransformDot<Value>(a, x, c);
  return m;
}

function TransformDot<Value>(a:Expression<Real[_]>, x:Value) ->
    TransformDot<Value> {
  return TransformDot<Value>(a, x, Boxed(0.0));
}
