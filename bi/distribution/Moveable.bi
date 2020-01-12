/*
 * Type-specific interface for delayed sampling $M$-path nodes with move
 * support.
 *
 * - Value: Value type.
 */
abstract class Moveable<Value> < Distribution<Value> {
  /**
   * Gradient.
   */
  dfdx:Value?;

  function setChild(child:Delay) {
    if !x? {
      if !this.child? {
        this.child <- child;
      } else {
        assert this.child! == child;
      }
    }
  }
  
  function grad(d:Value) -> Boolean {
    if dfdx? {
      dfdx <- dfdx! + d;
    } else {
      dfdx <- d;
    }
    return true;
  }
}


function simulate_propose(x:Real, d:Real) -> Real {
  return simulate_gaussian(x + 0.03*d, 0.06);
}

function simulate_propose(x:Real[_], d:Real[_]) -> Real[_] {
  return simulate_multivariate_gaussian(x + d, 1.0);
}

function simulate_propose(x:Real[_,_], d:Real[_,_]) -> Real[_,_] {
  return simulate_matrix_gaussian(x + d, 1.0);
}

function simulate_propose(x:Integer, d:Integer) -> Integer {
  return x;
}

function simulate_propose(x:Integer[_], d:Integer[_]) -> Integer[_] {
  return x;
}

function simulate_propose(x:Boolean, d:Boolean) -> Boolean {
  return x;
}

function logpdf_propose(x':Real, x:Real, d:Real) -> Real {
  return logpdf_gaussian(x', x + 0.03*d, 0.06);
}

function logpdf_propose(x':Real[_], x:Real[_], d:Real[_]) -> Real {
  return logpdf_multivariate_gaussian(x', x + d, 1.0);
}

function logpdf_propose(x':Real[_,_], x:Real[_,_], d:Real[_,_]) -> Real {
  return logpdf_matrix_gaussian(x', x + d, 1.0);
}

function logpdf_propose(x':Integer, x:Integer, d:Integer) -> Real {
  return 0.0;
}

function logpdf_propose(x':Integer[_], x:Integer[_], d:Integer[_]) -> Real {
  return 0.0;
}

function logpdf_propose(x':Boolean, x:Boolean, d:Boolean) -> Real {
  return 0.0;
}
