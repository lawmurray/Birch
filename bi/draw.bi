program draw(
    input_file:String <- "output/simulation.json",
    output_file:String <- "figs/plot.svg",
    width:Integer <- 1024,
    height:Integer <- 1024) {
  input:JSONReader <- JSONReader(input_file);
  surface:Surface <- createSVG(output_file, width, height);
  cr:Context <- create(surface);

  /* background */
  cr.setSourceRGB(1.0, 1.0, 1.0);
  cr.rectangle(0, 0, width, height);
  cr.fill();
  
  col:Real[_] <- [0.3373, 0.7059, 0.9137];
  
  auto l <- input.getRealVector(["θ", "l"])!;
  auto u <- input.getRealVector(["θ", "u"])!;

  auto scaleX <- width/(u[1] - l[1]);
  auto scaleY <- height/(u[2] - l[2]);
  auto scale <- max(scaleX, scaleY);
  auto fat <- 2.0;
  
  cr.scale(scaleX, scaleY);
  cr.translate(-l[1], -l[2]);

  /* clutter */
  cr.pushGroup();
  auto y <- input.getArray("y");
  while y? {
    auto Y <- y!.getRealMatrix();
    if Y? {
      cr.setSourceRGB(0.9, 0.9, 0.9);
      cr.setLineWidth(2.0*fat/scale);
      for i:Integer in 1..rows(Y!) {
        cr.arc(Y![i,1], Y![i,2], 4.0*fat/scale, 0.0, 2.0*π);
        cr.fill();
      }
    }
  }
  cr.popGroupToSource();
  cr.paint();

  /* tracks */
  auto z <- input.getArray("z");
  while z? {
    cr.pushGroup();
    auto X <- z!.getRealMatrix("x");
    if X? {
      if rows(X!) > 0 {
        cr.setLineWidth(6.0*fat/scale);
        cr.setSourceRGB(col[1], col[2], col[3]);
        cr.moveTo(X![1,1], X![1,2]);
        for i:Integer in 2..rows(X!) {
          cr.lineTo(X![i,1], X![i,2]);
        }
        cr.stroke();
      }
      for i:Integer in 1..rows(X!) {
        cr.arc(X![i,1], X![i,2], 6.0*fat/scale, 0.0, 2.0*π);
        cr.fill();
      }
    }

    cr.setLineWidth(2.0*fat/scale);
    auto ys <- z!.getArray("y");
    while ys? {
      auto y <- ys!.getRealVector();
      if y? {
        cr.arc(y![1], y![2], 6.0*fat/scale, 0.0, 2.0*π);
        cr.stroke();
      }
    }
    cr.popGroupToSource();
    cr.paint();
  }
  
  cr.destroy();
  surface.destroy();
}
