/**
 * Lazy access of a matrix element.
 */
final class MatrixElement<Single,Value>(Y:Single, i:Integer, j:Integer) <
    ScalarExpression<Value> {
  /**
   * Matrix.
   */
  Y:Single <- Y;
    
  /**
   * Row.
   */
  i:Integer <- i;

  /**
   * Column.
   */
  j:Integer <- j;
  
  override function doValue() {
    x <- matrix(Y.value())[i,j];
  }

  override function doGet() {
    x <- matrix(Y.get())[i,j];
  }

  override function doPilot() {
    x <- matrix(Y.pilot())[i,j];
  }

  override function doMove(κ:Kernel) {
    x <- matrix(Y.move(κ))[i,j];
  }

  override function doGrad() {
    Y.grad(d!, i, j);
  }

  final override function doMakeConstant() {
    Y.makeConstant();
  }

  final override function doRestoreCount() {
    Y.restoreCount();
  }
  
  final override function doPrior(vars:RaggedArray<DelayExpression>) ->
      Expression<Real>? {
    return Y.prior(vars);
  }
}

/**
 * Lazy access of a matrix element.
 */
function MatrixElement(Y:Expression<Real[_,_]>, i:Integer, j:Integer) ->
    Expression<Real> {
  if Y.isConstant() {
    return box(Y.value()[i,j]);
  } else {
    m:MatrixElement<Expression<Real[_,_]>,Real>(Y, i, j);
    return m;
  }
}

/**
 * Lazy access of a matrix element.
 */
function MatrixElement(Y:Expression<LLT>, i:Integer, j:Integer) ->
    Expression<Real> {
  if Y.isConstant() {
    return box(matrix(Y.value())[i,j]);
  } else {
    m:MatrixElement<Expression<LLT>,Real>(Y, i, j);
    return m;
  }
}

/**
 * Lazy access of a matrix element.
 */
function MatrixElement(Y:Expression<Integer[_,_]>, i:Integer, j:Integer) ->
    Expression<Integer> {
  if Y.isConstant() {
    return box(Y.value()[i,j]);
  } else {
    m:MatrixElement<Expression<Integer[_,_]>,Integer>(Y, i, j);
    return m;
  }
}
