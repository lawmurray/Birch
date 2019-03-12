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
 */
class StarModel<Parameter,Point> < StateModel<Parameter,Point> {  
  fiber simulate() -> Real {
    /* parameters */
    yield sum(parameter(θ));

    /* points */
    auto f <- this.x.walk();    
    x:Point?;  // current point
    while true {
      if f? {  // is the next point given?
        x <- f!;
      } else {
        x':Point;
        this.x.pushBack(x');
        x <- x';
      }
      yield sum(point(x!, θ));
    }
  }

  function checkpoints() -> Integer? {
    /* one checkpoint for the parameters, then one for each point */
    return 1 + x.size();
  }

  /**
   * Point model.
   */
  fiber point(x:Point, θ:Parameter) -> Real {
    //
  }
}
