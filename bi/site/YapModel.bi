/**
 * Model for Yap case study.
 */
class YapModel(T:Integer) {
  θ:YapParameter;
  x:YapState[T];
  
  /**
   * Run the model.
   *
   *   - T: number of time steps.
   */
  fiber run() -> Real! {
    θ.run();
    x[1].run(θ);
    for (t:Integer in 2..T) {
      x[t].run(x[t-1], θ);
    }
  }
}
