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
    auto u <- z!.getRealMatrix("y")!;
    
    for i:Integer in 1..rows(u) {
      v:Random<Real[_]>;
      v <- u[i,1..2];
      y.get(t + i - 1).pushBack(v);
    }
  }
  
  auto writer <- JSONWriter(output_file);
  y.write(writer.setObject("y"));
  θ.write(writer.setObject("θ"));
  writer.save();
}
