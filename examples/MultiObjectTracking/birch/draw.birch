/**
 * Draw a figure of simulate or filter results.
 *
 */
program draw(input:String <- "output/simulate.json",
    output:String <- "figs/simulate.pdf",
    width:Integer <- 1024,
    height:Integer <- 1024) {
  /* input file */
  auto reader <- Reader(input);
  auto inputBuffer <- reader.scan();
  reader.close();
  
  /* output file and drawing surface */
  auto surface <- createPDF(output, width, height);
  auto cr <- create(surface);

  /* background */
  cr.setSourceRGB(0.95, 0.95, 0.95);
  cr.rectangle(0, 0, width, height);
  cr.fill();

  /* config */
  //auto col <- [0.3373, 0.7059, 0.9137]; // blue
  auto col <- [0.8353, 0.3686, 0.0000];  // red

  auto array <- inputBuffer.walk();
  if array? {
    auto sample <- array!.getObject("sample");
    if sample? {
      auto θ <- sample!.getChild("θ");
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
      auto y <- sample!.walk("y");
      while y? {
        auto Y <- y!.getRealMatrix();
        if Y? {
          cr.setSourceRGB(0.8, 0.8, 0.8);
          for i in 1..rows(Y!) {
            cr.arc(Y![i,1], Y![i,2], 4.0*fat/scale, 0.0, 2.0*π);
            cr.fill();
          }
        }
      }

      /* circle those points indicating associated observations */
      auto z <- sample!.walk("z");
      while z? {
        cr.setLineWidth(2.0*fat/scale);
        auto ys <- z!.walk("y");
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
      cr.setSourceRGB(col[1], col[2], col[3]);
      z <- sample!.walk("z");
      while z? {
         auto x <- z!.getRealMatrix("x");
        if x? {
          cr.setLineWidth(4.0*fat/scale);
          cr.moveTo(x![1,1], x![1,2]);
          for i in 2..rows(x!) {
            cr.lineTo(x![i,1], x![i,2]);
          }
          cr.stroke();

          for i in 2..rows(x!) {
            cr.arc(x![i,1], x![i,2], 4.0*fat/scale, 0.0, 2.0*π);
            cr.fill();
          }
        }
      }
    
      /* start time labels for latent tracks */
      z <- sample!.walk("z");
      while z? {
        auto t <- z!.getInteger("t");
        auto x <- z!.getRealMatrix("x");
    
        if t? && x? {
          cr.setLineWidth(2.0*fat/scale);
          cr.setSourceRGB(col[1], col[2], col[3]);
          cr.arc(x![1,1], x![1,2], 10.0*fat/scale, 0.0, 2.0*π);
          cr.fill();

          cr.setSourceRGB(1.0, 1.0, 1.0);
          cr.setFontSize(0.5);
          cr.moveTo(x![1,1] - 0.26, x![1,2] + 0.15);
          cr.showText(String(t!));
        }
      }
    }
  }
  
  /* destroy the surface (triggers save) */
  cr.destroy();
  surface.destroy();
}
