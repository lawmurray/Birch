program draw(
    input_file:String <- "output/simulation.json",
    output_file:String <- "figs/simulation.pdf",
    width:Integer <- 1024,
    height:Integer <- 1024) {
  input:JSONReader <- JSONReader(input_file);
  surface:Surface <- createPDF(output_file, width, height);
  cr:Context <- create(surface);

  /* background */
  cr.setSourceRGB(0.95, 0.95, 0.95);
  cr.rectangle(0, 0, width, height);
  cr.fill();

  /* border */
  cr.setSourceRGB(0.8, 0.8, 0.8);
  cr.rectangle(0, 0, width - 1, height - 1);
  cr.stroke();

  
  //col:Real[_] <- [0.3373, 0.7059, 0.9137];
  col:Real[_] <- [0.8353, 0.3686, 0.0000];

  auto l <- input.getRealVector(["θ", "l"])!;
  auto u <- input.getRealVector(["θ", "u"])!;

  auto scaleX <- width/(u[1] - l[1]);
  auto scaleY <- height/(u[2] - l[2]);
  auto scale <- max(scaleX, scaleY);
  auto fat <- 2.0;
  
  cr.scale(scaleX, scaleY);
  cr.translate(-l[1], -l[2]);

  /* clutter */
  auto y <- input.getArray("y");
  while y? {
    auto Y <- y!.getRealMatrix();
    if Y? {
      cr.setSourceRGB(0.8, 0.8, 0.8);
      for i:Integer in 1..rows(Y!) {
        cr.arc(Y![i,1], Y![i,2], 4.0*fat/scale, 0.0, 2.0*π);
        cr.fill();
      }
    }
  }

  /* track observations */
  auto z <- input.getArray("z");
  while z? {
    cr.setLineWidth(2.0*fat/scale);
    auto ys <- z!.getArray("y");
    while ys? {
      auto y <- ys!.getRealVector();
      if y? {
        cr.setSourceRGB(0.8, 0.8, 0.8);
        cr.arc(y![1], y![2], 4.0*fat/scale, 0.0, 2.0*π);
        cr.fill();

        cr.setSourceRGB(col[1], col[2], col[3]);
        cr.arc(y![1], y![2], 4.0*fat/scale, 0.0, 2.0*π);
        cr.stroke();
      }
    }
  }
    
  /* track paths */
  z <- input.getArray("z");
  while z? {
    auto X <- z!.getRealMatrix("x");
    if X? {
      cr.setLineWidth(4.0*fat/scale);
      cr.moveTo(X![1,1], X![1,2]);
      for i:Integer in 2..rows(X!) {
        cr.lineTo(X![i,1], X![i,2]);
      }
      cr.stroke();

      for i:Integer in 2..rows(X!) {
        cr.arc(X![i,1], X![i,2], 4.0*fat/scale, 0.0, 2.0*π);
        cr.fill();
      }
    }
  }
    
  /* track labels */
  z <- input.getArray("z");
  while z? {
    auto t <- z!.getInteger("t")!;
    auto X <- z!.getRealMatrix("x");
    
    cr.setLineWidth(2.0*fat/scale);
    cr.setSourceRGB(col[1], col[2], col[3]);
    cr.arc(X![1,1], X![1,2], 10.0*fat/scale, 0.0, 2.0*π);
    cr.fill();
        
    cr.setSourceRGB(1.0, 1.0, 1.0);
    cr.setFontSize(0.5);
    cr.moveTo(X![1,1] - 0.26, X![1,2] + 0.15);
    cr.showText(String(t));
  }
  
  cr.destroy();
  surface.destroy();
}
