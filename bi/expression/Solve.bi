function solve(x:Expression<Real[_,_]>, y:Expression<Real[_]>) ->
    Expression<Real[_]> {
  return inv(x)*y;
}

function solve(x:Real[_,_], y:Expression<Real[_]>) -> Expression<Real[_]> {
  return inv(x)*y;
}

function solve(x:Expression<Real[_,_]>, y:Real[_]) -> Expression<Real[_]> {
  return inv(x)*y;
}
