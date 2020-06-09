/**
 * Lazy rank downdate.
 */
final class MatrixRankDowndate<Left,Right,Value>(left:Left, right:Right) <
    MatrixBinaryExpression<Left,Right,Value>(left, right) {  
  override function rows() -> Integer {
    return left.rows();
  }
  
  override function columns() -> Integer {
    return left.columns();
  }

  override function doValue() {
    x <- rank_downdate(left.value(), right.value());
  }

  override function doGet() {
    x <- rank_downdate(left.get(), right.get());
  }

  override function doPilot() {
    x <- rank_downdate(left.pilot(), right.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- rank_downdate(left.move(κ), right.move(κ));
  }

  override function doGrad() {
    left.grad(D!);
    right.grad(-(D! + transpose(D!))*right.get());
  }
}

/**
 * Lazy rank 1 downdate.
 */
function rank_downdate(left:Expression<LLT>, right:Expression<Real[_]>) ->
    Expression<LLT> {
  assert left.columns() == right.rows();
  if left.isConstant() && right.isConstant() {
    return box(rank_downdate(left.value(), right.value()));
  } else {
    m:MatrixRankDowndate<Expression<LLT>,Expression<Real[_]>,LLT>(left, right);
    return m;
  }
}

/**
 * Lazy rank 1 downdate.
 */
function rank_downdate(left:LLT, right:Expression<Real[_]>) ->
    Expression<LLT> {
  if right.isConstant() {
    return box(rank_downdate(left, right.value()));
  } else {
    return rank_downdate(box(left), right);
  }
}

/**
 * Lazy rank 1 downdate.
 */
function rank_downdate(left:Expression<LLT>, right:Real[_]) ->
    Expression<LLT> {
  if left.isConstant() {
    return box(rank_downdate(left.value(), right));
  } else {
    return rank_downdate(left, box(right));
  }
}

/**
 * Lazy rank $k$ downdate.
 */
function rank_downdate(left:Expression<LLT>, right:Expression<Real[_,_]>) ->
    Expression<LLT> {
  assert left.columns() == right.rows();
  if left.isConstant() && right.isConstant() {
    return box(rank_downdate(left.value(), right.value()));
  } else {
    m:MatrixRankDowndate<Expression<LLT>,Expression<Real[_,_]>,LLT>(left, right);
    return m;
  }
}

/**
 * Lazy rank $k$ downdate.
 */
function rank_downdate(left:LLT, right:Expression<Real[_,_]>) ->
    Expression<LLT> {
  if right.isConstant() {
    return box(rank_downdate(left, right.value()));
  } else {
    return rank_downdate(box(left), right);
  }
}

/**
 * Lazy rank $k$ downdate.
 */
function rank_downdate(left:Expression<LLT>, right:Real[_,_]) ->
    Expression<LLT> {
  if left.isConstant() {
    return box(rank_downdate(left.value(), right));
  } else {
    return rank_downdate(left, box(right));
  }
}
