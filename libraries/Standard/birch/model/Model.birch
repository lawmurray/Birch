/**
 * A model.
 *
 * ```mermaid
 * classDiagram
 *    Model <|-- MarkovModel
 *    Model <|-- HiddenMarkovModel
 *    HiddenMarkovModel -- StateSpaceModel
 *    link Model "../Model/"
 * ```
 */
abstract class Model {
  /**
   * Size. This is the number of steps of `simulate(Integer)` to be performed
   * after the initial call to `simulate()`.
   */
  function size() -> Integer {
    return 0;
  }

  /**
   * Simulate.
   */
  fiber simulate() -> Event {
    //
  }

  /**
   * Simulate the `t`th step.
   */
  fiber simulate(t:Integer) -> Event {
    //
  }

  /**
   * Forecast the `t`th step.
   */
  fiber forecast(t:Integer) -> Event {
    //
  }
}
