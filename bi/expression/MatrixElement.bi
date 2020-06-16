/**
 * Lazy access of a matrix element.
 */
final class MatrixElement<Base,RowIndex,ColumnIndex>(Y:Base, i:RowIndex,
    j:ColumnIndex) < ScalarExpression<Real> {
  /**
   * Matrix.
   */
  Y:Base <- Y;
    
  /**
   * Row.
   */
  i:RowIndex <- i;

  /**
   * Column.
   */
  j:ColumnIndex <- j;
  
  override function doValue() {
    x <- Real(matrix(Y.value())[i.value(), j.value()]);
  }

  override function doGet() {
    x <- Real(matrix(Y.get())[i.get(), j.get()]);
  }

  override function doPilot() {
    x <- Real(matrix(Y.pilot())[i.pilot(), j.pilot()]);
  }

  override function doMove(κ:Kernel) {
    x <- Real(matrix(Y.move(κ))[i.move(κ), j.move(κ)]);
  }

  override function doGrad() {
    Y.grad(d!, i.get(), j.get());
    i.grad(0.0);
    j.grad(0.0);
  }

  final override function doMakeConstant() {
    Y.makeConstant();
    i.makeConstant();
    j.makeConstant();
  }

  final override function doRestoreCount() {
    Y.restoreCount();
    i.restoreCount();
    j.restoreCount();
  }
  
  final override function doPrior(vars:RaggedArray<DelayExpression>) ->
      Expression<Real>? {
    r:Expression<Real>?;   
     
    auto p1 <- Y.prior(vars);
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
    
    auto p3 <- j.prior(vars);
    if p3? {
      if r? {
        r <- p3! + r!;
      } else {
        r <- p3;
      }
    }
    
    return r;
  }
}

/**
 * Lazy access of a matrix element.
 */
function MatrixElement(Y:Expression<Real[_,_]>, i:Expression<Integer>,
    j:Expression<Integer>) -> Expression<Real> {
  if Y.isConstant() && i.isConstant() && j.isConstant() {
    return box(Real(Y.value()[i.value(), j.value()]));
  } else {
    m:MatrixElement<Expression<Real[_,_]>,Expression<Integer>,Expression<Integer>>(Y, i, j);
    return m;
  }
}

/**
 * Lazy access of a matrix element.
 */
function MatrixElement(Y:Expression<LLT>, i:Expression<Integer>,
    j:Expression<Integer>) -> Expression<Real> {
  if Y.isConstant() && i.isConstant() && j.isConstant() {
    return box(Real(matrix(Y.value())[i.value(), j.value()]));
  } else {
    m:MatrixElement<Expression<LLT>,Expression<Integer>,Expression<Integer>>(Y, i, j);
    return m;
  }
}

/**
 * Lazy access of a matrix element.
 */
function MatrixElement(Y:Expression<Integer[_,_]>, i:Expression<Integer>,
    j:Expression<Integer>) -> Expression<Real> {
  if Y.isConstant() && i.isConstant() && j.isConstant() {
    return box(Real(Y.value()[i.value(), j.value()]));
  } else {
    m:MatrixElement<Expression<Integer[_,_]>,Expression<Integer>,Expression<Integer>>(Y, i, j);
    return m;
  }
}
