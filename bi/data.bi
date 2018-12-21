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
  inputBuffer:JSONBuffer;
  inputBuffer.load(input);
  
  /* read in the simulated observations from the track */
  θ:Global;
  y:Vector<List<Random<Real[_]>>>;
  
  inputBuffer.get("θ", θ);  
  inputBuffer.get("y", y);
  
  auto z <- inputBuffer.getArray("z");
  while z? {
    auto t <- z!.getInteger("t")!;
    auto u <- z!.getArray("y");
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
  
  /* save the observations to the output file */
  outputBuffer:JSONBuffer;
  outputBuffer.set("y", y);
  outputBuffer.set("θ", θ);
  outputBuffer.save(output);
}
