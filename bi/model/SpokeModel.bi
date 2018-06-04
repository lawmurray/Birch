/**
 * Hub and spoke model, structured as the parameters (the hub) and
 * any number of additional models that are conditionally independent
 * given the parameters (the spokes).
 *
 * The joint distribution is:
 *
 * $$p(\mathrm{d}x_{1:N}, \mathrm{d}\theta) = p(\mathrm{d}\theta)
 *   \prod_{n=1}^N p(\mathrm{d}x_n \mid \theta)$$
 *
 * <center>
 * ![Graphical model depiction of SpokeModel.](../figs/SpokeModel.svg)
 * </center>
 */
class SpokeModel<SpokeType,ParameterType> < Model {
  /**
   * Hub.
   */
  θ:ParameterType;

  /**
   * Spokes.
   */
  spokes:List<SpokeType>;

  fiber simulate() -> Real {
    /* hub */
    θ.parameter();

    /* spokes */
    f:SpokeType! <- spokes.walk();
    while (f?) {
      f!.simulate(θ);
    }
  }
  
  function input(reader:Reader) {
    r:Reader? <- reader.getObject("parameter");
    if (r?) {
      θ.input(r!);
    }
    
    r <- reader.getObject("spokes");
    if (!r?) {
      /* try root instead */
      r <- reader;
    }
    f:Reader! <- r!.getArray();
    while (f?) {
      x:SpokeType;
      x.input(f!);
      spokes.pushBack(x);
    }
  }
  
  function output(writer:Writer) {
    w:Writer <- writer.setObject("parameter");
    θ.output(w);
    
    w <- writer.setArray("spokes");
    f:SpokeType! <- spokes.walk();
    while (f?) {
      f!.output(w.push());
    }
  }
}
