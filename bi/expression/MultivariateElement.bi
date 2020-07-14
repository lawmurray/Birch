/**
 * Lazy access of a vector element.
 */
final class MultivariateElement<Value>(y:Expression<Value[_]>, i:Integer) <
    ScalarExpression<Value> {
  /**
   * Argument.
   */
  y:Expression<Value[_]>? <- y;
    
  /**
   * Element.
   */
  i:Integer <- i;

  override function doDepth() -> Integer {
    return y!.depth() + 1;
  }

  override function doValue() -> Value {
    return y!.value()[i];
  }

  override function doPilot() -> Value {
    return y!.pilot()[i];
  }

  override function doGet() -> Value {
    return y!.get()[i];
  }

  override function doMove(κ:Kernel) -> Value {
    return y!.move(κ)[i];
  }

  override function doGrad() {
    y!.grad(d!, i);
  }

  override function doPrior(vars:RaggedArray<DelayExpression>) ->
      Expression<Real>? {
    return y!.prior(vars);
  }

  override function doConstant() {
    y!.constant();
  }

  override function doCount() {
    y!.count();
  }

  override function doDetach() {
    y <- nil;
  }
}

/**
 * Lazy access of a vector element.
 */
function MultivariateElement(y:Expression<Real[_]>, i:Integer) ->
    MultivariateElement<Real> {
  return construct<MultivariateElement<Real>>(y, i);
}

/**
 * Lazy access of a vector element.
 */
function MultivariateElement(y:Expression<Integer[_]>, i:Integer) ->
    MultivariateElement<Integer> {
  return construct<MultivariateElement<Integer>>(y, i);
}

/**
 * Lazy access of a vector element.
 */
function MultivariateElement(y:Expression<Boolean[_]>, i:Integer) ->
    MultivariateElement<Boolean> {
  return construct<MultivariateElement<Boolean>>(y, i);
}
