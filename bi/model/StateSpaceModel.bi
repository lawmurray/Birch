/**
 * State-space model.
 *
 * The joint distribution is:
 *
 * $$p(\mathrm{d}\theta, \mathrm{d}x_{0:T}, \mathrm{d}y_{0:T}) =
 *   p(\mathrm{d}\theta) p(\mathrm{d}x_0 \mid \theta)  p(\mathrm{d}y_0
 *   \mid x_0, \theta) \prod_{t=1}^T p(\mathrm{d}x_t \mid x_{t-1},
 *   \theta) p(\mathrm{d}y_t \mid x_t, \theta)$$
 */
class StateSpaceModel<Parameter,State,Observation> <
    MarkovModel<Parameter,State> {
  /**
   * Observation history.
   */
  y:List<Observation>;
  
  /**
   * Observation future.
   */
  y1:List<Observation>;

  function start() -> Real {
    auto w <- super.start();

    /* initial observation */
    if (!y1.empty()) {
      y':Observation <- y1.front();
      y1.popFront();
      w <- w + sum(observation(y', x.back(), θ));
      y.pushBack(y');
    } else {
      y':Observation;
      w <- w + sum(observation(y', x.back(), θ));
      y.pushBack(y');
    }

    return w;
  }

  function step() -> Real {
    auto w <- super.step();

    /* observation */
    if (!y1.empty()) {
      y':Observation <- y1.front();
      y1.popFront();
      w <- w + sum(observation(y', x.back(), θ));
      y.pushBack(y');
    } else {
      y':Observation;
      w <- w + sum(observation(y', x.back(), θ));
      y.pushBack(y');
    }

    return w;
  }

  function read(buffer:Buffer) {
    super.read(buffer);
    buffer.get("y", y1);
  }
  
  function write(buffer:Buffer) {
    super.write(buffer);
    buffer.set("y", y);
  }
  
  /**
   * Observation model.
   */
  fiber observation(x':Observation, x:State, θ:Parameter) -> Real {
    //
  }
}
