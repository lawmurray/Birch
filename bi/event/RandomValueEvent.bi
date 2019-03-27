/**
 * Event triggered by a distribution being attached to a random variate with
 * the `~` operator.
 *
 * - v: The random variate.
 * - p: The distribution.
 */
class RandomValueEvent<Value>(v:Random<Value>, p:Distribution<Value>) <
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
    v.value();
  }

  function value(evt:Event) {
    auto r <- RandomValueEvent<Value>?(evt);
    if r? {
      v <- r!.v.value();
    } else {
      error("incompatible traces");
    }
  }
}
