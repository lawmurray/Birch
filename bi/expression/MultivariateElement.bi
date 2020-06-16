/**
 * Lazy access of a vector element.
 */
final class MultivariateElement<Base,Index>(y:Base, i:Index) <
    ScalarExpression<Real> {
  /**
   * Vector.
   */
  y:Base <- y;
    
  /**
   * Element.
   */
  i:Index <- i;
  
  override function doValue() {
    x <- y.value()[i.value()];
  }

  override function doGet() {
    x <- y.get()[i.get()];
  }

  override function doPilot() {
    x <- y.pilot()[i.pilot()];
  }

  override function doMove(κ:Kernel) {
    x <- y.move(κ)[i.move(κ)];
  }

  override function doGrad() {
    y.grad(d!, i.get());
    i.grad(0.0);
  }

  final override function doMakeConstant() {
    y.makeConstant();
    i.makeConstant();
  }

  final override function doRestoreCount() {
    y.restoreCount();
    i.restoreCount();
  }
  
  final override function doPrior(vars:RaggedArray<DelayExpression>) ->
      Expression<Real>? {
    r:Expression<Real>?;   
     
    auto p1 <- y.prior(vars);
    if p1? {
      if r? {
        r <- p1! + r!;
      } else {
        r <- p1;
      }
    }
    
    auto p2 <- i.prior(vars);
    if p2? {
      if r? {
        r <- p2! + r!;
      } else {
        r <- p2;
      }
    }
    
    return r;
  }
}

/**
 * Lazy access of a vector element.
 */
function MultivariateElement(y:Expression<Real[_]>,
    i:Expression<Integer>) -> Expression<Real> {
  if y.isConstant() && i.isConstant() {
    return box(Real(y.value()[i.value()]));
  } else {
    m:MultivariateElement<Expression<Real[_]>,Expression<Integer>>(y, i);
    return m;
  }
}

/**
 * Lazy access of a vector element.
 */
function MultivariateElement(y:Expression<Integer[_]>,
    i:Expression<Integer>) -> Expression<Real> {
  if y.isConstant() && i.isConstant() {
    return box(Real(y.value()[i.value()]));
  } else {
    m:MultivariateElement<Expression<Integer[_]>,Expression<Integer>>(y, i);
    return m;
  }
}
