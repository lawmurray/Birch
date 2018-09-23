/**
 * Base class for all models with a specific variate type.
 */
class ModelFor<SpecificVariate> < Model {
  /**
   * Simulate.
   *
   * - v: The variate. This is cast to `SpecificVariate` and passed through
   *      to the more-specific overload.
   */
  fiber simulate(v:Variate) -> Real {
    auto w <- SpecificVariate?(v);
    if (!w?) {
      stderr.print("incorrect specific variate type");
      exit(1);
    }
    simulate(w!);
  }

  /**
   * Simulate.
   *
   * - v: The variate.
   *
   * This overload of `simulate` should be overridden by derived classes.
   */
  fiber simulate(v:SpecificVariate) -> Real {
    //
  }
}