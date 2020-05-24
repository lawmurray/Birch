/**
 * Lazy multiply.
 */
final class DiscreteMultiply<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  override function computeValue(l:Left, r:Right) -> Value {
    return l*r;
  }
  
  override function computeGrad(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d*r, d*l);
  }

  override function graftDiscrete() -> Discrete? {
    r:Discrete? <- graftBoundedDiscrete();
    if !r? {
      /* match a template */
      x:Discrete?;
      if (x <- left.graftDiscrete())? {
        r <- LinearDiscrete(right, x!, Boxed(0));
      } else if (x <- right.graftDiscrete())? {
        r <- LinearDiscrete(left, x!, Boxed(0));
      }
    }
    return r;
  }

  override function graftBoundedDiscrete() -> BoundedDiscrete? {
    x1:BoundedDiscrete? <- left.graftBoundedDiscrete();
    x2:BoundedDiscrete? <- right.graftBoundedDiscrete();
    r:BoundedDiscrete?;

    /* match a template */       
    if x1? {
      r <- LinearBoundedDiscrete(right, x1!, Boxed(0));
    } else if x2? {
      r <- LinearBoundedDiscrete(left, x2!, Boxed(0));
    }
    
    return r;
  }
}

/**
 * Lazy add.
 */
operator (left:Expression<Integer>*right:Expression<Integer>) ->
    Expression<Integer> {
  if left.isConstant() && right.isConstant() {
    return box(left.value() + right.value());
  } else {
    m:DiscreteMultiply<Integer,Integer,Integer>(left, right);
    return m;
  }
}

/**
 * Lazy add.
 */
operator (left:Integer*right:Expression<Integer>) -> Expression<Integer> {
  if right.isConstant() {
    return box(left + right.value());
  } else {
    return Boxed(left)*right;
  }
}

/**
 * Lazy add.
 */
operator (left:Expression<Integer>*right:Integer) -> Expression<Integer> {
  if left.isConstant() {
    return box(left.value() + right);
  } else {
    return left*Boxed(right);
  }
}
