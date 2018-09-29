program data(
    input_file:String <- "output/simulation.json",
    output_file:String <- "input/filter.json") {
  auto reader <- JSONReader(input_file);
  
  θ:Global;
  θ.read(reader.getObject("θ"));
  
  y:Vector<List<Random<Real[_]>>>;
  y.read(reader.getObject("y"));
  
  auto z <- reader.getArray("z");
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
  
  auto writer <- JSONWriter(output_file);
  y.write(writer.setObject("y"));
  θ.write(writer.setObject("θ"));
  writer.save();
}
