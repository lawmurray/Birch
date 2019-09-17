/*
 * Lazy multivariate addition.
 */
final class MultivariateAdd<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < Expression<Value> {  
  /**
   * Left operand.
   */
  left:Expression<Left> <- left;
  
  /**
   * Right operand.
   */
  right:Expression<Right> <- right;

  function value() -> Value {
    return left.value() + right.value();
  }

  function graftLinearMultivariateGaussian() ->
      TransformLinearMultivariate<DelayMultivariateGaussian>? {
    y:TransformLinearMultivariate<DelayMultivariateGaussian>?;
    z:DelayMultivariateGaussian?;

    if (y <- left.graftLinearMultivariateGaussian())? {
      y!.add(right.value());
    } else if (y <- right.graftLinearMultivariateGaussian())? {
      y!.add(left.value());
    } else if (z <- left.graftMultivariateGaussian())? {
      y <- TransformLinearMultivariate<DelayMultivariateGaussian>(
          identity(z!.size()), z!, right.value());
    } else if (z <- right.graftMultivariateGaussian())? {
      y <- TransformLinearMultivariate<DelayMultivariateGaussian>(
          identity(z!.size()), z!, left.value());
    }
    return y;
  }
  
  function graftLinearIdenticalNormalInverseGamma() ->
      TransformLinearMultivariate<DelayIdenticalNormalInverseGamma>? {
    y:TransformLinearMultivariate<DelayIdenticalNormalInverseGamma>?;
    z:DelayIdenticalNormalInverseGamma?;

    if (y <- left.graftLinearIdenticalNormalInverseGamma())? {
      y!.add(right.value());
    } else if (y <- right.graftLinearIdenticalNormalInverseGamma())? {
      y!.add(left.value());
    } else if (z <- left.graftIdenticalNormalInverseGamma())? {
      y <- TransformLinearMultivariate<DelayIdenticalNormalInverseGamma>(
          identity(z!.size()), z!, right.value());
    } else if (z <- right.graftIdenticalNormalInverseGamma())? {
      y <- TransformLinearMultivariate<DelayIdenticalNormalInverseGamma>(
          identity(z!.size()), z!, left.value());
    }
    return y;
  }
}

operator (left:Expression<Real[_]> + right:Expression<Real[_]>) ->
    MultivariateAdd<Real[_],Real[_],Real[_]> {
  m:MultivariateAdd<Real[_],Real[_],Real[_]>(left, right);
  return m;
}

operator (left:Real[_] + right:Expression<Real[_]>) ->
    MultivariateAdd<Real[_],Real[_],Real[_]> {
  return Boxed(left) + right;
}

operator (left:Expression<Real[_]> + right:Real[_]) ->
    MultivariateAdd<Real[_],Real[_],Real[_]> {
  return left + Boxed(right);
}
