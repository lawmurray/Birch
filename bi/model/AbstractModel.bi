/**
 * Base class for all models. This enables runtime loading of model classes.
 */
class AbstractModel {
  /**
   * Make a variate for use with this model.
   */
  function variate() -> Object {
    //
  }

  /**
   * Simulate.
   */
  fiber simulate(v:Object) -> Real {
    //
  }
}
