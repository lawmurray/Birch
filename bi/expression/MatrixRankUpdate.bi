/**
 * Lazy rank update.
 */
final class MatrixRankUpdate<Left,Right,Value>(left:Left, right:Right) <
    MatrixBinaryExpression<Left,Right,Value>(left, right) {  
  override function doRows() -> Integer {
    return left!.rows();
  }
  
  override function doColumns() -> Integer {
    return left!.columns();
  }

  override function doValue() {
    x <- rank_update(left!.value(), right!.value());
  }

  override function doPilot() {
    x <- rank_update(left!.pilot(), right!.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- rank_update(left!.move(κ), right!.move(κ));
  }

  override function doGrad() {
    left!.grad(d!);
    right!.grad((d! + transpose(d!))*right!.get());
  }
}

/**
 * Lazy rank 1 update.
 */
function rank_update(left:Expression<LLT>, right:Expression<Real[_]>) ->
    Expression<LLT> {
  assert left!.columns() == right!.rows();
  if left!.isConstant() && right!.isConstant() {
    return box(rank_update(left!.value(), right!.value()));
  } else {
    return construct<MatrixRankUpdate<Expression<LLT>,Expression<Real[_]>,LLT>>(left, right);
  }
}

/**
 * Lazy rank 1 update.
 */
function rank_update(left:LLT, right:Expression<Real[_]>) ->
    Expression<LLT> {
  if right!.isConstant() {
    return box(rank_update(left, right!.value()));
  } else {
    return rank_update(box(left), right);
  }
}

/**
 * Lazy rank 1 update.
 */
function rank_update(left:Expression<LLT>, right:Real[_]) ->
    Expression<LLT> {
  if left!.isConstant() {
    return box(rank_update(left!.value(), right));
  } else {
    return rank_update(left, box(right));
  }
}

/**
 * Lazy rank $k$ update.
 */
function rank_update(left:Expression<LLT>, right:Expression<Real[_,_]>) ->
    Expression<LLT> {
  assert left!.columns() == right!.rows();
  if left!.isConstant() && right!.isConstant() {
    return box(rank_update(left!.value(), right!.value()));
  } else {
    return construct<MatrixRankUpdate<Expression<LLT>,Expression<Real[_,_]>,LLT>>(left, right);
  }
}

/**
 * Lazy rank $k$ update.
 */
function rank_update(left:LLT, right:Expression<Real[_,_]>) ->
    Expression<LLT> {
  if right!.isConstant() {
    return box(rank_update(left, right!.value()));
  } else {
    return rank_update(box(left), right);
  }
}

/**
 * Lazy rank $k$ update.
 */
function rank_update(left:Expression<LLT>, right:Real[_,_]) ->
    Expression<LLT> {
  if left!.isConstant() {
    return box(rank_update(left!.value(), right));
  } else {
    return rank_update(left, box(right));
  }
}
