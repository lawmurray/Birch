/**
 * Lazy `join`.
 */
final class Join<Value>(x:Expression<Value>[_]) < Expression<Value[_]> {
  /**
   * Arguments.
   */
  args:Expression<Value>[_] <- x;

  override function rows() -> Integer {
    return global.length(args);
  }

  final override function doValue() {
    x <- transform(args, @(x:Expression<Value>) -> Value {
        return x.value();
      });
  }

  final override function doMakeConstant() {
    for_each(args, @(x:Expression<Value>) { x.makeConstant(); });
  }
  
  final override function doPilot() {
    x <- transform(args, @(x:Expression<Value>) -> Value {
        return x.pilot();
      });
  }

  final override function doRestoreCount() {
    for_each(args, @(x:Expression<Value>) { x.restoreCount(); });
  }

  final override function doMove(κ:Kernel) {
    x <- transform(args, @(x:Expression<Value>) -> Value {
        return x.move(κ);
      });
  }
  
  final override function doGrad() {
    for_each(args, dfdx!, @(x:Expression<Value>, d:Value) { x.grad(d); });
  }

  final override function doPrior(vars:RaggedArray<DelayExpression>) ->
      Expression<Real>? {
    p:Expression<Real>?;
    for i in 1..global.length(args) {
      auto q <- args[i].prior(vars);
      if q? {
        if p? {
          p <- p! + q!;
        } else {
          p <- q;
        }
      }
    }
    return p;
  }
}

/**
 * Lazy `join`. Converts a vector of scalar expressions into a single vector
 * expression.
 */
function join(x:Expression<Real>[_]) -> Expression<Real[_]> {
  m:Join<Real>(x);
  return m;
}
