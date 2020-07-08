/**
 * Lazy `join`.
 */
final class MatrixJoin<Value>(X:Expression<Value>[_,_]) <
    MatrixExpression<Value[_,_]> {
  /**
   * Arguments.
   */
  args:Expression<Value>[_,_]? <- X;

  override function doRows() -> Integer {
    return global.rows(args!);
  }

  override function doColumns() -> Integer {
    return global.columns(args!);
  }

  override function doDetach() {
    args <- nil;
  }

  override function doValue() {
    x <- transform(args!, \(x:Expression<Value>) -> Value {
        return x.value();
      });
  }

  override function doMakeConstant() {
    for_each(args!, \(x:Expression<Value>) { x.makeConstant(); });
  }
  
  override function doPilot() {
    x <- transform(args!, \(x:Expression<Value>) -> Value {
        return x.pilot();
      });
  }

  override function doMove(κ:Kernel) {
    x <- transform(args!, \(x:Expression<Value>) -> Value {
        return x.move(κ);
      });
  }
  
  override function doGrad() {
    for_each(args!, d!, \(x:Expression<Value>, d:Value) { x.grad(d); });
  }

  override function doPrior(vars:RaggedArray<DelayExpression>) ->
      Expression<Real>? {
    p:Expression<Real>?;
    auto R <- rows();
    auto C <- columns();
    for i in 1..R {
      for j in 1..C {
        auto q <- args![i,j].prior(vars);
        if q? {
          if p? {
            p <- p! + q!;
          } else {
            p <- q;
          }
        }
      }
    }
    return p;
  }
}

/**
 * Lazy `join`. Converts a matrix of scalar expressions into a single matrix
 * expression.
 */
function join(X:Expression<Real>[_,_]) -> Expression<Real[_,_]> {
  return construct<MatrixJoin<Real>>(X);
}

/**
 * Lazy `split`. Converts a matrix expression into a matrix of scalar
 * expressions.
 */
function split(X:Expression<Real[_,_]>) -> Expression<Real>[_,_] {
  return matrix(\(i:Integer, j:Integer) -> Expression<Real> {
        return MatrixElement(matrix(X), i, j);
      }, X.rows(), X.columns());
  // ^ matrix(X) above is an identity function for all but Random objects;
  //   for these it wraps the Random in an additional expression that can
  //   accumulate gradients by element (which a Random cannot) before passing
  //   the whole matrix of accumulated gradients onto the Random
}

/**
 * Lazy `split`. Converts a matrix expression into a matrix of scalar
 * expressions.
 */
function split(X:Expression<LLT>) -> Expression<Real>[_,_] {
  return matrix(\(i:Integer, j:Integer) -> Expression<Real> {
        return MatrixElement(matrix(X), i, j);
      }, X.rows(), X.columns());
}
