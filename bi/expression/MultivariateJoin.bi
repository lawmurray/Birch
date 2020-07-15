/**
 * Lazy `join`.
 */
final class MultivariateJoin<Value>(x:Expression<Value>[_]) <
    MultivariateExpression<Value[_]> {
  /**
   * Arguments.
   */
  y:Expression<Value>[_]? <- x;

  override function doDepth() -> Integer {
    auto depth <- 0;
    for i in 1..length() {
      depth <- max(depth, y![i].depth());
    }
    return depth + 1;
  }

  override function doRows() -> Integer {
    return global.length(y!);
  }

  override function doValue() -> Value[_] {
    return transform(y!, \(x:Expression<Value>) -> Value {
        return x.value();
      });
  }
  
  override function doPilot(gen:Integer) -> Value[_] {
    return transform(y!, \(x:Expression<Value>) -> Value {
        return x.pilot(gen);
      });
  }

  override function doGet() -> Value[_] {
    return transform(y!, \(x:Expression<Value>) -> Value {
        return x.get();
      });
  }

  override function doMove(gen:Integer, κ:Kernel) -> Value[_] {
    return transform(y!, \(x:Expression<Value>) -> Value {
        return x.move(gen, κ);
      });
  }
  
  override function doGrad(gen:Integer) {
    for_each(y!, d!, \(x:Expression<Value>, d:Value) { x.grad(gen, d); });
  }

  override function doPrior(vars:RaggedArray<DelayExpression>) ->
      Expression<Real>? {
    p:Expression<Real>?;
    auto L <- length();
    for i in 1..L {
      auto q <- y![i].prior(vars);
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

  override function doCount(gen:Integer) {
    for_each(y!, \(x:Expression<Value>) { x.count(gen); });
  }

  override function doConstant() {
    for_each(y!, \(x:Expression<Value>) { x.constant(); });
  }

  override function doDetach() {
    y <- nil;
  }
}

/**
 * Lazy `join`. Converts a vector of scalar expressions into a vector
 * expression.
 */
function join<Value>(y:Expression<Value>[_]) -> MultivariateJoin<Value> {
  return construct<MultivariateJoin<Value>>(y);
}

/**
 * Lazy `split`. Converts a vector expression into a vector of scalar
 * expressions.
 */
function split<Value>(y:Expression<Value[_]>) -> Expression<Value>[_] {
  auto z <- vector(y);
  // ^ vector(y) above is an identity function for all but Random objects;
  //   for these it wraps the Random in an additional expression that can
  //   accumulate gradients by element (which a Random cannot) before passing
  //   the whole vector of accumulated gradients onto the Random

  return vector(\(i:Integer) -> Expression<Value> {
        return construct<MultivariateElement<Value>>(z, i);
      }, z.length());
}
