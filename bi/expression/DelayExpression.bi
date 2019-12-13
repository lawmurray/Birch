/**
 * Expression that wraps a delayed sampling node.
 *
 * - Value: Value type.
 */
final class DelayExpression<Value>(delay:DelayValue<Value>) <
    Expression<Value> {  
  /**
   * Delayed sampling node.
   */
  delay:DelayValue<Value>& <- delay;
  
  function graft() -> Expression<Value> {
    return this;
  }

  function setChild(child:Delay) {
    delay.setChild(child);
  }

  function value() -> Value {
    return delay.value();
  }
  
  function pilot() -> Value {
    return delay.pilot();
  }
  
  function propose() -> Value {
    return delay.propose();
  }
  
  function gradPilot(d:Value) -> Boolean {
    return delay.gradPilot(d);
  }

  function gradPropose(d:Value) -> Boolean {
    return delay.gradPropose(d);
  }
  
  function ratio() -> Real {
    return delay.ratio();
  }
  
  function accept() {
    delay.accept();
  }

  function reject() {
    delay.reject();
  }
  
  function clamp() {
    delay.clamp();
  }
}

/**
 * Create a DelayExpression.
 */
function DelayExpression<Value>(delay:DelayValue<Value>) -> DelayExpression<Value> {
  o:DelayExpression<Value>(delay);
  return o;
}
