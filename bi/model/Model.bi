/**
 * Model.
 */
class Model<Variate> < AbstractModel {   
  /**
   * Simulate.
   *
   * - v: The variate.
   */
  fiber simulate(v:Variate) -> Real {
    //
  }

  function variate() -> AbstractVariate {
    v:Variate;
    return v;
  }

  fiber simulate(v:AbstractVariate) -> Real {
    auto w <- Variate?(v);
    if (w?) {
      simulate(w!);
    } else {
      
    }
  }
}
