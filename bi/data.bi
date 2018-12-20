program data(
    input_file:String <- "output/simulation.json",
    output_file:String <- "input/filter.json") {
  input:JSONBuffer;
  input.load(input_file);
  
  θ:Global;
  y:Vector<List<Random<Real[_]>>>;
  
  input.get("θ", θ);  
  input.get("y", y);
  
  auto z <- input.getArray("z");
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
  
  output:JSONBuffer;
  output.set("y", y);
  output.set("θ", θ);
  output.save(output_file);
}
