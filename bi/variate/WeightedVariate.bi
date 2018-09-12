/**
 * Variate with a weight.
 */
class WeightedVariate<Variate>(x:Variate, w:Real) {
  /**
   * The variate.
   */
  x:Variate <- x;

  /**
   * Log-weight of the variate.
   */
  w:Real <- w;

  function read(reader:Reader) {
    reader.getObject("x")!.read(x);
    w <- reader.getReal("w")!;
  }
  
  function write(writer:Writer) {
    writer.setObject("x").write(x);
    writer.setReal("w", w);
  }
}
