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
  
  palette:Real[_,_] <- [[0.3373, 0.7059, 0.9137],
                        [0.8353, 0.3686, 0.0000],
                        [0.0000, 0.6196, 0.4510],
                        [0.9020, 0.6235, 0.0000],
                        [0.8000, 0.4745, 0.6549],
                        [0.9412, 0.8941, 0.2588],
                        [0.0000, 0.4471, 0.6980]];
  
  auto l <- input.getRealVector(["θ", "l"])!;
  auto u <- input.getRealVector(["θ", "u"])!;

  auto scaleX <- width/(u[1] - l[1]);
  auto scaleY <- height/(u[2] - l[2]);
  auto scale <- max(scaleX, scaleY);
  
  cr.scale(scaleX, scaleY);
  cr.translate(-l[1], -l[2]);

  /* clutter */
  auto y <- input.getArray("y");
  while y? {
    auto Y <- y!.getRealMatrix();
    if Y? {
      cr.setSourceRGB(0.9, 0.9, 0.9);
      cr.setLineWidth(2.0/scale);
      for i:Integer in 1..rows(Y!) {
        cr.arc(Y![i,1], Y![i,2], 4.0/scale, 0.0, 2.0*π);
        cr.fill();
      }
    }
  }

  /* tracks */
  auto z <- input.getArray("z");
  auto col <- 1;
  while z? {
    auto X <- z!.getRealMatrix("x");
    if (X? && rows(X!) > 0) {
      cr.setLineWidth(4.0/scale);
      cr.setSourceRGB(palette[col,1], palette[col,2], palette[col,3]);
      cr.moveTo(X![1,1], X![1,2]);
      for i:Integer in 2..rows(X!) {
        cr.lineTo(X![i,1], X![i,2]);
      }
      cr.stroke();

      for i:Integer in 1..rows(X!) {
        cr.arc(X![i,1], X![i,2], 6.0/scale, 0.0, 2.0*π);
        cr.fill();
      }
    }

    auto Y <- z!.getRealMatrix("y");
    if (Y? && rows(Y!) > 0) {
      cr.setLineWidth(2.0/scale);
      for i:Integer in 1..rows(Y!) {
        cr.arc(Y![i,1], Y![i,2], 6.0/scale, 0.0, 2.0*π);
        cr.stroke();
      }
    }
    
    col <- mod(col, rows(palette)) + 1;
  }
  
  cr.destroy();
  surface.destroy();
}
