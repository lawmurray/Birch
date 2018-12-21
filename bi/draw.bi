/**
 * Draw a figure of simulate or filter results.
 *
 */
program draw(input:String <- "output/simulate.json",
    output:String <- "figs/simulate.pdf",
    width:Integer <- 1024,
    height:Integer <- 1024) {
  /* input file */
  inputBuffer:JSONBuffer;
  inputBuffer.load(input);
  
  /* output file and drawing surface */
  surface:Surface <- createPDF(output, width, height);
  cr:Context <- create(surface);

  /* background */
  cr.setSourceRGB(0.95, 0.95, 0.95);
  cr.rectangle(0, 0, width, height);
  cr.fill();

  /* config */
  //col:Real[_] <- [0.3373, 0.7059, 0.9137]; // blue
  col:Real[_] <- [0.8353, 0.3686, 0.0000];  // red

  auto θ <- inputBuffer.getChild("θ");
  auto l <- θ!.getRealVector("l")!;
  auto u <- θ!.getRealVector("u")!;

  auto scaleX <- width/(u[1] - l[1]);
  auto scaleY <- height/(u[2] - l[2]);
  auto scale <- max(scaleX, scaleY);
  auto fat <- 2.0;
  
  /* border */
  cr.setSourceRGB(0.8, 0.8, 0.8);
  cr.rectangle(0, 0, width - 1, height - 1);
  cr.stroke();

  /* set scale for tracking domain */
  cr.scale(scaleX, scaleY);
  cr.translate(-l[1], -l[2]);

  /* solid points indicating clutter */
  auto y <- inputBuffer.getArray("y");
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

  /* circle those points indicating associated observations */
  auto z <- inputBuffer.getArray("z");
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
    
  /* lines and points marking latent tracks */
  z <- inputBuffer.getArray("z");
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
    
  /* start time labels for latent tracks */
  z <- inputBuffer.getArray("z");
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
  
  /* destroy the surface (triggers save) */
  cr.destroy();
  surface.destroy();
}
