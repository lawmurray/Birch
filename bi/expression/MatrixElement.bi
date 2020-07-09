/**
 * Lazy access of a matrix element.
 */
final class MatrixElement<Single,Value>(Y:Single, i:Integer, j:Integer) <
    ScalarExpression<Value> {
  /**
   * Matrix.
   */
  Y:Single? <- Y;
    
  /**
   * Row.
   */
  i:Integer <- i;

  /**
   * Column.
   */
  j:Integer <- j;
  
  override function depth() -> Integer {
    auto argDepth <- 0;
    if Y? {
      argDepth <- Y!.depth();
    }
    return argDepth + 1;
  }
  
  override function doDetach() {
    Y <- nil;
  }
  
  override function doValue() {
    x <- matrix(Y!.value())[i,j];
  }

  override function doPilot() {
    x <- matrix(Y!.pilot())[i,j];
  }

  override function doMove(κ:Kernel) {
    x <- matrix(Y!.move(κ))[i,j];
  }

  override function doGrad() {
    Y!.grad(d!, i, j);
  }

  final override function doMakeConstant() {
    Y!.makeConstant();
  }
  
  final override function doPrior(vars:RaggedArray<DelayExpression>) ->
      Expression<Real>? {
    return Y!.prior(vars);
  }
}

/**
 * Lazy access of a matrix element.
 */
function MatrixElement(Y:Expression<Real[_,_]>, i:Integer, j:Integer) ->
    Expression<Real> {
  if Y!.isConstant() {
    return box(Y.value()[i,j]);
  } else {
    return construct<MatrixElement<Expression<Real[_,_]>,Real>>(Y, i, j);
  }
}

/**
 * Lazy access of a matrix element.
 */
function MatrixElement(Y:Expression<LLT>, i:Integer, j:Integer) ->
    Expression<Real> {
  if Y!.isConstant() {
    return box(matrix(Y.value())[i,j]);
  } else {
    return construct<MatrixElement<Expression<LLT>,Real>>(Y, i, j);
  }
}

/**
 * Lazy access of a matrix element.
 */
function MatrixElement(Y:Expression<Integer[_,_]>, i:Integer, j:Integer) ->
    Expression<Integer> {
  if Y!.isConstant() {
    return box(Y.value()[i,j]);
  } else {
    return construct<MatrixElement<Expression<Integer[_,_]>,Integer>>(Y, i, j);
  }
}
