program draw(
    input_file:String <- "output/simulation.json",
    output_file:String <- "figs/plot.png",
    width:Integer <- 1024,
    height:Integer <- 1024) {
  input:JSONReader <- JSONReader(input_file);
  surface:Surface <- createPNG(output_file, width, height);
  cr:Context <- create(surface);

  /* background */
  cr.setSourceRGB(1.0, 1.0, 1.0);
  cr.rectangle(0, 0, width, height);
  cr.fill();
  
  auto l <- input.getRealVector(["θ", "l"])!;
  auto u <- input.getRealVector(["θ", "u"])!;

  auto scaleX <- width/(u[1] - l[1]);
  auto scaleY <- height/(u[2] - l[2]);
  auto scale <- max(scaleX, scaleY);
  
  cr.scale(scaleX, scaleY);
  cr.translate(-l[1], -l[2]);
  cr.setLineWidth(2.0/scale);

  /* clutter */
  auto y <- input.getArray("y");
  while y? {
    auto Y <- y!.getRealMatrix();
    if Y? {
      cr.setSourceRGB(0.8, 0.8, 0.8);
      for i:Integer in 1..rows(Y!) {
        cr.arc(Y![i,1], Y![i,2], 4.0/scale, 0.0, 2.0*π);
        cr.fill();
      }
    }
  }

  /* tracks */
  auto z <- input.getArray("z");
  while z? {
    auto X <- z!.getRealMatrix("x");
    if (X? && rows(X!) > 0) {
      cr.setSourceRGB(0.0, 0.0, 0.0);
      cr.arc(X![1,1], X![1,2], 8.0/scale, 0.0, 2.0*π);
      cr.stroke();
      cr.moveTo(X![1,1], X![1,2]);
      for i:Integer in 2..rows(X!) {
        cr.lineTo(X![i,1], X![i,2]);
      }
      cr.stroke();
    }

    auto Y <- z!.getRealMatrix("y");
    if (Y? && rows(Y!) > 0) {
      cr.setSourceRGB(0.0, 0.0, 0.0);
      for i:Integer in 1..rows(Y!) {
        cr.arc(Y![i,1], Y![i,2], 4.0/scale, 0.0, 2.0*π);
        cr.fill();
      }
    }
  }
  
  cr.destroy();
  surface.destroy();
}
