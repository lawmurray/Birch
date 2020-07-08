/**
 * Lazy multivariate solve.
 */
final class MultivariateSolve<Left,Right,Value>(left:Left, right:Right) <
    MultivariateBinaryExpression<Left,Right,Value>(left, right) {  
  override function doRows() -> Integer {
    return left!.rows();
  }
  
  override function doValue() {
    x <- solve(left!.value(), right!.value());
  }

  override function doPilot() {
    x <- solve(left!.pilot(), right!.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- solve(left!.move(κ), right!.move(κ));
  }

  override function doGrad() {
    auto L <- left!.get();
    auto r <- right!.get();
    left!.grad(-solve(transpose(L), d!)*transpose(solve(L, r)));
    right!.grad(solve(transpose(L), d!));
  }
}

/**
 * Lazy solve.
 */
function solve(left:Expression<Real[_,_]>, right:Expression<Real[_]>) ->
    Expression<Real[_]> {
  assert left!.columns() == right!.rows();
  if left!.isConstant() && right!.isConstant() {
    return box(vector(solve(left!.value(), right!.value())));
  } else {
    return construct<MultivariateSolve<Expression<Real[_,_]>,Expression<Real[_]>,Real[_]>>(left, right);
  }
}

/**
 * Lazy solve.
 */
function solve(left:Real[_,_], right:Expression<Real[_]>) ->
    Expression<Real[_]> {
  if right!.isConstant() {
    return box(vector(solve(left, right!.value())));
  } else {
    return solve(box(left), right);
  }
}

/**
 * Lazy solve.
 */
function solve(left:Expression<Real[_,_]>, right:Real[_]) ->
    Expression<Real[_]> {
  if left!.isConstant() {
    return box(vector(solve(left!.value(), right)));
  } else {
    return solve(left, box(right));
  }
}

/**
 * Lazy solve.
 */
function solve(left:Expression<LLT>, right:Expression<Real[_]>) ->
    Expression<Real[_]> {
  assert left!.columns() == right!.rows();
  if left!.isConstant() && right!.isConstant() {
    return box(vector(solve(left!.value(), right!.value())));
  } else {
    return construct<MultivariateSolve<Expression<LLT>,Expression<Real[_]>,Real[_]>>(left, right);
  }
}

/**
 * Lazy solve.
 */
function solve(left:LLT, right:Expression<Real[_]>) -> Expression<Real[_]> {
  if right!.isConstant() {
    return box(vector(solve(left, right!.value())));
  } else {
    return solve(box(left), right);
  }
}

/**
 * Lazy solve.
 */
function solve(left:Expression<LLT>, right:Real[_]) -> Expression<Real[_]> {
  if left!.isConstant() {
    return box(vector(solve(left!.value(), right)));
  } else {
    return solve(left, box(right));
  }
}
