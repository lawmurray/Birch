/**
 * Create a simulated data set from simulation output.
 *
 * `--input`: path of an output file from a simulation run, containing the
 * ground truth.
 * `--output`: path to an output file to which to write the simulated data
 * set.
 */
program data(input:String <- "output/simulate.json",
    output:String <- "input/filter.json") {
  /* input */
  inputBuffer:MemoryBuffer;
  inputBuffer.load(input);
  
  /* read in the simulated observations from the track */
  θ:Global;
  y:Vector<Vector<Random<Real[_]>>>;

  auto array <- inputBuffer.walk();
  if array? {
    auto sample <- array!.getObject("sample");
    if sample? {
      sample!.get("θ", θ);  
      sample!.get("y", y);
      auto z <- sample!.walk("z");
      while z? {
        auto t <- z!.getInteger("t")!;
        auto u <- z!.walk("y");
        while u? {
          auto v <- u!.getRealVector();
          if v? {
            w:Random<Real[_]>;
            w <- v![1..2];
            y.get(t).pushBack(w);
          }
          t <- t + 1;
        }
      }
    }
  }
  
  /* save the observations to the output file */
  outputBuffer:MemoryBuffer;
  outputBuffer.set("y", y);
  outputBuffer.set("θ", θ);
  outputBuffer.save(output);
}
