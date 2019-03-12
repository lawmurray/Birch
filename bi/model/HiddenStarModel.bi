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
 */
class HiddenStarModel<Parameter,Point,Observation> <
    ObservedModel<Parameter,Point,Observation> {
  fiber simulate() -> Real {
    /* parameters */
    yield sum(parameter(θ));
    
    /* points and observations */
    auto f <- this.x.walk();
    auto g <- this.y.walk();
    
    x:Point?;        // current point
    y:Observation?;  // current observation
    
    while true {
      w:Real <- 0.0;
      if f? {  // is the next point given?
        x <- f!;
      } else {
        x':Point;
        this.x.pushBack(x');
        x <- x';
      }
	  w <- sum(point(x!, θ));
      
      if g? {  // is the next observation given?
        y <- g!;
      } else {
        y':Observation;
        this.y.pushBack(y');
        y <- y';
      }
      w <- w + sum(observation(y!, x!, θ));
      yield w;
    }
  }

  function checkpoints() -> Integer? {
    /* one checkpoint for the parameters, then one for each point */
    return 1 + max(x.size(), y.size());
  }

  /**
   * Point model.
   */
  fiber point(x:Point, θ:Parameter) -> Real {
    //
  }

  /**
   * Observation model.
   */
  fiber observation(y:Observation, x:Point, θ:Parameter) -> Real {
    //
  }
}
