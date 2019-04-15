/**
 * Event triggered by a distribution being attached to a random variate with
 * the `~` operator.
 *
 * - v: The random variate.
 * - p: The distribution.
 */
final class RandomValueEvent<Value>(v:Random<Value>, p:Distribution<Value>) <
    RandomEvent {
  /**
   * Random variable associated with the event.
   */
  v:Random<Value> <- v;
  
  /**
   * Distribution associated with the event.
   */
  p:Distribution<Value> <- p;

  function accept(h:EventHandler) -> Real {
    return h.handle(this);
  }

  function hasValue() -> Boolean {
    return v.hasValue();
  }

  function hasDistribution() -> Boolean {
    return v.hasDistribution();
  }

  function assume() {
    v.assume(p);
  }
  
  function observe() -> Real {
    assert hasValue();
    return p.observe(v);
  }

  function value() {
    v.assume(p);
    v.value();
  }

  function downdate(evt:Event) {
    assert hasValue();
    return p.downdate(v);
  }

  function assumeUpdate(evt:Event) {
    v.assumeUpdate(p, cast(evt).v.value());
  }

  function assumeDowndate(evt:Event) {
    v.assumeDowndate(p, cast(evt).v.value());
  }

  function value(evt:Event) {
    v.assume(p);
    v <- cast(evt).v.value();
  }
  
  /**
   * Cast `evt` to this type, giving an error if not possible.
   */
  function cast(evt:Event) -> RandomValueEvent<Value> {
    auto r <- RandomValueEvent<Value>?(evt);
    if r? {
      return r!;
    } else {
      error("incompatible traces");
    }
  }
}
