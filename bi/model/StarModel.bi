/**
 * Star model.
 *
 * The joint distribution is:
 *
 * $$p(\mathrm{d}\theta, \mathrm{d}x_{1:N}) = p(\mathrm{d}\theta) 
 *   \prod_{n=1}^N p(\mathrm{d}x_n \mid \theta).$$
 */
class StarModel<Parameter,Point> < Model {
  /**
   * Parameter.
   */
  θ:Parameter;

  /**
   * Points.
   */
  x:List<Point>;
  
  fiber simulate() -> Real {
    /* parameter */
    yield sum(parameter(θ));
    
    /* points */
    auto x <- this.x.walk();
    while (x?) {
      point(x!, θ);
    }
  }

  /**
   * Parameter model.
   */
  fiber parameter(θ:Parameter) -> Real {
    //
  }
    
  /**
   * Point model.
   */
  fiber point(x:Point, θ:Parameter) -> Real {
    //
  }

  function read(buffer:Buffer) {
    buffer.get("θ", θ);
    buffer.get("x", x);
  }
  
  function write(buffer:Buffer) {
    buffer.set("θ", θ);
    buffer.set("x", x);
  }
}
