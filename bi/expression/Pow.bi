/*
 * Lazy `pow`.
 */
final class Pow<Left,Right,Value>(x:Expression<Left>, y:Expression<Right>) <
    Expression<Value> {  
  /**
   * Base.
   */
  x:Expression<Left> <- x;
  
  /**
   * Exponent.
   */
  y:Expression<Right> <- y;

  function value() -> Value {
    return pow(x.value(), y.value());
  }

  function pilot() -> Value {
    return pow(x.pilot(), y.pilot());
  }
  
  function grad(d:Value) {
    x.grad(d*y.pilot()*pow(x.pilot(), y.pilot() - 1));
    y.grad(d*pow(x.pilot(), y.pilot())*log(x.pilot()));
  }
}

function pow(x:Expression<Real>, y:Expression<Real>) -> Pow<Real,Real,Real> {
  m:Pow<Real,Real,Real>(x, y);
  return m;
}

function pow(x:Real, y:Expression<Real>) -> Pow<Real,Real,Real> {
  return pow(Boxed(x), y);
}

function pow(x:Expression<Real>, y:Real) -> Pow<Real,Real,Real> {
  return pow(x, Boxed(y));
}
