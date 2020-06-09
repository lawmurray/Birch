/**
 * Lazy `join`.
 */
final class Join<Value>(x:Expression<Value>[_]) <
    MultivariateExpression<Value[_]> {
  /**
   * Arguments.
   */
  args:Expression<Value>[_] <- x;

  override function rows() -> Integer {
    return global.length(args);
  }

  override function doValue() {
    x <- transform(args, @(x:Expression<Value>) -> Value {
        return x.value();
      });
  }

  override function doMakeConstant() {
    for_each(args, @(x:Expression<Value>) { x.makeConstant(); });
  }

  override function doGet() {
    x <- transform(args, @(x:Expression<Value>) -> Value {
        return x.get();
      });
  }
  
  override function doPilot() {
    x <- transform(args, @(x:Expression<Value>) -> Value {
        return x.pilot();
      });
  }

  override function doRestoreCount() {
    for_each(args, @(x:Expression<Value>) { x.restoreCount(); });
  }

  override function doMove(κ:Kernel) {
    x <- transform(args, @(x:Expression<Value>) -> Value {
        return x.move(κ);
      });
  }
  
  override function doGrad() {
    for_each(args, d!, @(x:Expression<Value>, d:Value) { x.grad(d); });
  }

  override function doPrior(vars:RaggedArray<DelayExpression>) ->
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
