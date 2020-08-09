/**
 * Lazy rank update.
 */
final class MatrixRankUpdate<Right,RightValue,RightGradient>(y:Expression<LLT>,
    z:Right) < MatrixBinaryExpression<Expression<LLT>,Right,LLT,RightValue,
    Real[_,_],RightGradient,LLT>(y, z) {  
  override function doRows() -> Integer {
    return y!.rows();
  }
  
  override function doColumns() -> Integer {
    return y!.columns();
  }

  override function doEvaluate(y:LLT, z:RightValue) -> LLT {
    return rank_update(y, z);
  }

  override function doEvaluateGradLeft(d:Real[_,_], x:LLT, y:LLT,
      z:RightValue) -> Real[_,_] {
    return d;
  }
  
  override function doEvaluateGradRight(d:Real[_,_], x:LLT, y:LLT,
      z:RightValue) -> RightGradient {
    return (d + transpose(d))*z;
  }
}

/**
 * Lazy rank 1 update.
 */
function rank_update(y:Expression<LLT>, z:Expression<Real[_]>) ->
    MatrixRankUpdate<Expression<Real[_]>,Real[_],Real[_]> {
  assert y!.columns() == z!.rows();
  return construct<MatrixRankUpdate<Expression<Real[_]>,Real[_],Real[_]>>(y, z);
}

/**
 * Lazy rank 1 update.
 */
function rank_update(y:LLT, z:Expression<Real[_]>) ->
    MatrixRankUpdate<Expression<Real[_]>,Real[_],Real[_]> {
  return rank_update(box(y), z);
}

/**
 * Lazy rank 1 update.
 */
function rank_update(y:Expression<LLT>, z:Real[_]) ->
    MatrixRankUpdate<Expression<Real[_]>,Real[_],Real[_]> {
  return rank_update(y, box(z));
}

/**
 * Lazy rank $k$ update.
 */
function rank_update(y:Expression<LLT>, z:Expression<Real[_,_]>) ->
    MatrixRankUpdate<Expression<Real[_,_]>,Real[_,_],Real[_,_]> {
  assert y!.columns() == z!.rows();
  return construct<MatrixRankUpdate<Expression<Real[_,_]>,Real[_,_],
      Real[_,_]>>(y, z);
}

/**
 * Lazy rank $k$ update.
 */
function rank_update(y:LLT, z:Expression<Real[_,_]>) ->
    MatrixRankUpdate<Expression<Real[_,_]>,Real[_,_],Real[_,_]> {
  return rank_update(box(y), z);
}

/**
 * Lazy rank $k$ update.
 */
function rank_update(y:Expression<LLT>, z:Real[_,_]) ->
    MatrixRankUpdate<Expression<Real[_,_]>,Real[_,_],Real[_,_]> {
  return rank_update(y, box(z));
}
