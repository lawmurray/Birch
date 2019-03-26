/**
 * Hidden star model.
 *
 * The joint distribution is:
 *
 * $$p(\mathrm{d}\theta, \mathrm{d}x_{1:T}, \mathrm{d}y_{1:T}) =
 * p(\mathrm{d}\theta) \prod_{t=1}^T p(\mathrm{d}x_t \mid \theta)
 * p(\mathrm{d}y_t \mid x_t, \theta).$$
 *
 * Typically the $Y_{1:T}$ are observed, although they need not be.
 *
 * <center>
 * ![Graphical model depicting HiddenStarModel.](../figs/HiddenStarModel.svg)
 * </center>
 *
 * A model inheriting from `HiddenStarModel` overrides the `parameter`,
 * `point`, and `observation` member fibers to specify the individual
 * components of the joint distribution.
 */
class HiddenStarModel<Parameter,Point,Observation> <
    StarModel<Parameter,Point> {
  /**
   * Observations.
   */
  y:Vector<Observation>;

  /**
   * Observation model.
   *
   * - y: The observation, to be set.
   * - x: The point.
   * - θ: The parameters.
   */
  fiber observation(y:Observation, x:Point, θ:Parameter) -> Event {
    //
  }

  /**
   * Play one step. Simulates through the next point and observation.
   */
  function step() -> Real {
    auto w <- super.step();
    return w + h.handle(observation(y.get(t), x.get(t), θ));
  }

  function seek(t:Integer) {
    super.seek(t);
    while x.size() < t {
      y:Observation;
      this.y.pushBack(y);
    }
  }

  function size() -> Integer {
    return max(x.size(), y.size());
  }

  function read(buffer:Buffer) {
    super.read(buffer);
    buffer.get("y", y);
  }
  
  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("y", y);
  }
}
