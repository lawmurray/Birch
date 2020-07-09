/**
 * Lazy `join`.
 */
final class MultivariateJoin<Value>(x:Expression<Value>[_]) <
    MultivariateExpression<Value[_]> {
  /**
   * Arguments.
   */
  args:Expression<Value>[_]? <- x;

  override function depth() -> Integer {
    auto argsDepth <- 0;
    if args? {
      for i in 1..length() {
        argsDepth <- max(argsDepth, args![i].depth());
      }
    }
    return argsDepth + 1;
  }

  override function doRows() -> Integer {
    return global.length(args!);
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
    auto n <- length();
    for i in 1..n {
      auto q <- args![i].prior(vars);
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
  return construct<MultivariateJoin<Real>>(x);
}

/**
 * Lazy `split`. Converts a vector expression into a vector of scalar
 * expressions.
 */
function split(x:Expression<Real[_]>) -> Expression<Real>[_] {
  return vector(\(i:Integer) -> Expression<Real> {
        return MultivariateElement(vector(x), i);
      }, x.length());
  // ^ vector(x) above is an identity function for all but Random objects;
  //   for these it wraps the Random in an additional expression that can
  //   accumulate gradients by element (which a Random cannot) before passing
  //   the whole vector of accumulated gradients onto the Random
}
