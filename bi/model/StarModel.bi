/**
 * Star model.
 *
 * The joint distribution is:
 *
 * $$p(\mathrm{d}\theta, \mathrm{d}x_{1:T}) = p(\mathrm{d}\theta) 
 *   \prod_{t=1}^T p(\mathrm{d}x_t \mid \theta).$$
 *
 * <center>
 * ![Graphical model depicting StarModel.](../figs/StarModel.svg)
 * </center>
 *
 * A model inheriting from `StarModel` overrides the `parameter` and `point`
 * member fibers to specify the individual components of the joint
 * distribution.
 */
class StarModel<Parameter,Point> < RandomAccessModel {
  /**
   * Parameter.
   */
  θ:Parameter;

  /**
   * Points.
   */
  x:Vector<Point>;

  /**
   * Parameter model.
   *
   * - θ: The parameters, to be set.
   */
  fiber parameter(θ:Parameter) -> Event {
    //
  }

  /**
   * Point model.
   *
   * - x: The point, to be set.
   * - θ: The parameters.
   */
  fiber point(x:Point, θ:Parameter) -> Event {
    //
  }

  /**
   * Start. Simulates through the parameter model.
   */
  function start() -> Real {
    return super.start() + h.handle(parameter(θ));
  }

  /**
   * Step. Simulates the next point.
   */
  function step() -> Real {
    return super.step() + h.handle(point(x.get(t), θ));
  }

  function seek(t:Integer) {
    super.seek(t);
    while x.size() < t {
      x:Point;
      this.x.pushBack(x);
    }
  }

  function size() -> Integer {
    return x.size();
  }

  fiber simulate() -> Event {
    parameter(θ);
    while true {
      step();
    }
  }

  function read(buffer:Buffer) {
    super.read(buffer);
    buffer.get("θ", θ);
    buffer.get("x", x);
  }
  
  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("θ", θ);
    buffer.set("x", x);
  }
}
