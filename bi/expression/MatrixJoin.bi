/**
 * Lazy `join`.
 */
final class MatrixJoin<Value>(y:Expression<Value>[_,_]) <
    MatrixExpression<Value[_,_]> {
  /**
   * Arguments.
   */
  y:Expression<Value>[_,_]? <- y;

  override function doDepth() -> Integer {
    auto depth <- 0;
    for i in 1..rows() {
      for j in 1..columns() {
        depth <- max(depth, y![i,j].depth());
      }
    }
    return depth + 1;
  }

  override function doRows() -> Integer {
    return global.rows(y!);
  }

  override function doColumns() -> Integer {
    return global.columns(y!);
  }

  override function doValue() {
    x <- transform(y!, \(x:Expression<Value>) -> Value {
        return x.value();
      });
  }
  
  override function doPilot() {
    x <- transform(y!, \(x:Expression<Value>) -> Value {
        return x.pilot();
      });
  }

  override function doGet() {
    x <- transform(y!, \(x:Expression<Value>) -> Value {
        return x.get();
      });
  }

  override function doMove(κ:Kernel) {
    x <- transform(y!, \(x:Expression<Value>) -> Value {
        return x.move(κ);
      });
  }
  
  override function doGrad() {
    for_each(y!, d!, \(x:Expression<Value>, d:Value) { x.grad(d); });
  }

  override function doPrior(vars:RaggedArray<DelayExpression>) ->
      Expression<Real>? {
    p:Expression<Real>?;
    auto R <- rows();
    auto C <- columns();
    for i in 1..R {
      for j in 1..C {
        auto q <- y![i,j].prior(vars);
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

  override function doCount() {
    for_each(y!, \(x:Expression<Value>) { x.count(); });
  }

  override function doConstant() {
    for_each(y!, \(x:Expression<Value>) { x.constant(); });
  }

  override function doDetach() {
    y <- nil;
  }
}

/**
 * Lazy `join`. Converts a matrix of scalar expressions into a matrix
 * expression.
 */
function join<Value>(y:Expression<Value>[_,_]) -> MatrixJoin<Value> {
  return construct<MatrixJoin<Value>>(y);
}

/**
 * Lazy `split`. Converts a matrix expression into a matrix of scalar
 * expressions.
 */
function split<Value>(y:Expression<Value[_,_]>) -> Expression<Value>[_,_] {
  auto z <- canonical(y);
  // ^ canonical(y) above is an identity function for all but Random objects;
  //   for these it wraps the Random in an additional expression that can
  //   accumulate gradients by element (which a Random cannot) before passing
  //   the whole matrix of accumulated gradients onto the Random

  return matrix(\(i:Integer, j:Integer) -> Expression<Value> {
        return construct<MatrixElement<Value>>(z, i, j);
      }, z.rows(), z.columns());
}
