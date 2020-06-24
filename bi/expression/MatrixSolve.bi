/**
 * Lazy matrix solve.
 */
final class MatrixSolve<Left,Right,Value>(left:Left, right:Right) <
    MatrixBinaryExpression<Left,Right,Value>(left, right) {  
  override function doRows() -> Integer {
    return left!.rows();
  }
  
  override function doColumns() -> Integer {
    return right!.columns();
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
    auto R <- right!.get();
    left!.grad(-solve(transpose(L), d!)*transpose(solve(L, R)));
    right!.grad(solve(transpose(L), d!));
  }
}

/**
 * Lazy solve.
 */
function solve(left:Expression<Real[_,_]>, right:Expression<Real[_,_]>) ->
    Expression<Real[_,_]> {
  assert left!.columns() == right!.rows();
  if left!.isConstant() && right!.isConstant() {
    return box(matrix(solve(left!.value(), right!.value())));
  } else {
    m:MatrixSolve<Expression<Real[_,_]>,Expression<Real[_,_]>,Real[_,_]>(left, right);
    return m;
  }
}

/**
 * Lazy solve.
 */
function solve(left:Real[_,_], right:Expression<Real[_,_]>) ->
    Expression<Real[_,_]> {
  if right!.isConstant() {
    return box(matrix(solve(left, right!.value())));
  } else {
    return solve(box(left), right);
  }
}

/**
 * Lazy solve.
 */
function solve(left:Expression<Real[_,_]>, right:Real[_,_]) ->
    Expression<Real[_,_]> {
  if left!.isConstant() {
    return box(matrix(solve(left!.value(), right)));
  } else {
    return solve(left, box(right));
  }
}

/**
 * Lazy solve.
 */
function solve(left:Expression<Real[_,_]>, right:Expression<LLT>) ->
    Expression<Real[_,_]> {
  assert left!.columns() == right!.rows();
  if left!.isConstant() && right!.isConstant() {
    return box(matrix(solve(left!.value(), right!.value())));
  } else {
    m:MatrixSolve<Expression<Real[_,_]>,Expression<LLT>,Real[_,_]>(left, right);
    return m;
  }
}

/**
 * Lazy solve.
 */
function solve(left:Real[_,_], right:Expression<LLT>) ->
    Expression<Real[_,_]> {
  if right!.isConstant() {
    return box(matrix(solve(left, right!.value())));
  } else {
    return solve(box(left), right);
  }
}

/**
 * Lazy solve.
 */
function solve(left:Expression<Real[_,_]>, right:LLT) ->
    Expression<Real[_,_]> {
  if left!.isConstant() {
    return box(matrix(solve(left!.value(), right)));
  } else {
    return solve(left, box(right));
  }
}

/**
 * Lazy solve.
 */
function solve(left:Expression<LLT>, right:Expression<Real[_,_]>) ->
    Expression<Real[_,_]> {
  assert left!.columns() == right!.rows();
  if left!.isConstant() && right!.isConstant() {
    return box(matrix(solve(left!.value(), right!.value())));
  } else {
    m:MatrixSolve<Expression<LLT>,Expression<Real[_,_]>,Real[_,_]>(left, right);
    return m;
  }
}

/**
 * Lazy solve.
 */
function solve(left:LLT, right:Expression<Real[_,_]>) -> Expression<Real[_,_]> {
  if right!.isConstant() {
    return box(matrix(solve(left, right!.value())));
  } else {
    return solve(box(left), right);
  }
}

/**
 * Lazy solve.
 */
function solve(left:Expression<LLT>, right:Real[_,_]) -> Expression<Real[_,_]> {
  if left!.isConstant() {
    return box(matrix(solve(left!.value(), right)));
  } else {
    return solve(left, box(right));
  }
}

/**
 * Lazy solve.
 */
function solve(left:Expression<LLT>, right:Expression<LLT>) ->
    Expression<Real[_,_]> {
  assert left!.columns() == right!.rows();
  if left!.isConstant() && right!.isConstant() {
    return box(matrix(solve(left!.value(), right!.value())));
  } else {
    m:MatrixSolve<Expression<LLT>,Expression<LLT>,Real[_,_]>(left, right);
    return m;
  }
}

/**
 * Lazy solve.
 */
function solve(left:LLT, right:Expression<LLT>) -> Expression<Real[_,_]> {
  if right!.isConstant() {
    return box(matrix(solve(left, right!.value())));
  } else {
    return solve(box(left), right);
  }
}

/**
 * Lazy solve.
 */
function solve(left:Expression<LLT>, right:LLT) -> Expression<Real[_,_]> {
  if left!.isConstant() {
    return box(matrix(solve(left!.value(), right)));
  } else {
    return solve(left, box(right));
  }
}
