/**
 * Cairo pattern.
 */
class Pattern {
  hpp{{
  cairo_pattern_t* pattern;
  }}

  function addColorStopRGB(offset:Real, red:Real, green:Real, blue:Real) {
    cpp{{
    cairo_pattern_add_color_stop_rgb(pattern, offset_, red_, green_, blue_);
    }}
  }

  function addColorStopRGBA(offset:Real, red:Real, green:Real, blue:Real,
      alpha:Real) {
    cpp{{
    cairo_pattern_add_color_stop_rgba(pattern, offset_, red_, green_, blue_,
        alpha_);
    }}
  }
  
  function destroy() {
    cpp{{
    cairo_pattern_destroy(pattern);
    }}
  }
}

function createRGB(red:Real, green:Real, blue:Real) -> Pattern {
  result:Pattern;
  cpp{{
  result_->pattern = cairo_pattern_create_rgb(red_, green_, blue_);
  }}
  return result;
}

function createRGBA(red:Real, green:Real, blue:Real, alpha:Real) -> Pattern {
  result:Pattern;
  cpp{{
  result_->pattern = cairo_pattern_create_rgba(red_, green_, blue_, alpha_);
  }}
  return result;
}

function createLinear(x0:Real, y0:Real, x1:Real, y1:Real) -> Pattern {
  result:Pattern;
  cpp{{
  result_->pattern = cairo_pattern_create_linear(x0_, y0_, x1_, y1_);
  }}
  return result;
}

function createRadial(cx0:Real, cy0:Real, radius0:Real, cx1:Real, cy1:Real,
    radius1:Real) -> Pattern {
  result:Pattern;
  cpp{{
  result_->pattern = cairo_pattern_create_radial(cx0_, cy0_, radius0_, cx1_,
      cy1_, radius1_);
  }}
  return result;
}
