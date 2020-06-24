/**
 * Lazy access of a vector element.
 */
final class MultivariateElement<Single,Value>(y:Single, i:Integer) <
    ScalarExpression<Value> {
  /**
   * Vector.
   */
  y:Single? <- y;
    
  /**
   * Element.
   */
  i:Integer <- i;

  override function doDetach() {
    y <- nil;
  }
  
  override function doValue() {
    x <- y!.value()[i];
  }

  override function doPilot() {
    x <- y!.pilot()[i];
  }

  override function doMove(κ:Kernel) {
    x <- y!.move(κ)[i];
  }

  override function doGrad() {
    y!.grad(d!, i);
  }

  final override function doMakeConstant() {
    y!.makeConstant();
  }
  
  final override function doPrior(vars:RaggedArray<DelayExpression>) ->
      Expression<Real>? {
    return y!.prior(vars);
  }
}

/**
 * Lazy access of a vector element.
 */
function MultivariateElement(y:Expression<Real[_]>, i:Integer) ->
    Expression<Real> {
  if y.isConstant() {
    return box(y.value()[i]);
  } else {
    m:MultivariateElement<Expression<Real[_]>,Real>(y, i);
    return m;
  }
}

/**
 * Lazy access of a vector element.
 */
function MultivariateElement(y:Expression<Integer[_]>, i:Integer) ->
    Expression<Integer> {
  if y.isConstant() {
    return box(y.value()[i]);
  } else {
    m:MultivariateElement<Expression<Integer[_]>,Integer>(y, i);
    return m;
  }
}
