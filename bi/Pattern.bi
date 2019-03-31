/**
 * Cairo pattern.
 */
class Pattern {
  hpp{{
  cairo_pattern_t* pattern;
  }}

  function addColorStopRGB(offset:Real, red:Real, green:Real, blue:Real) {
    cpp{{
    cairo_pattern_add_color_stop_rgb(self->pattern, offset, red, green, blue);
    }}
  }

  function addColorStopRGBA(offset:Real, red:Real, green:Real, blue:Real,
      alpha:Real) {
    cpp{{
    cairo_pattern_add_color_stop_rgba(self->pattern, offset, red, green, blue,
        alpha);
    }}
  }
  
  function destroy() {
    cpp{{
    cairo_pattern_destroy(self->pattern);
    }}
  }
}

function createRGB(red:Real, green:Real, blue:Real) -> Pattern {
  result:Pattern;
  cpp{{
  result->pattern = cairo_pattern_create_rgb(red, green, blue);
  }}
  return result;
}

function createRGBA(red:Real, green:Real, blue:Real, alpha:Real) -> Pattern {
  result:Pattern;
  cpp{{
  result->pattern = cairo_pattern_create_rgba(red, green, blue, alpha);
  }}
  return result;
}

function createLinear(x0:Real, y0:Real, x1:Real, y1:Real) -> Pattern {
  result:Pattern;
  cpp{{
  result->pattern = cairo_pattern_create_linear(x0, y0, x1, y1);
  }}
  return result;
}

function createRadial(cx0:Real, cy0:Real, radius0:Real, cx1:Real, cy1:Real,
    radius1:Real) -> Pattern {
  result:Pattern;
  cpp{{
  result->pattern = cairo_pattern_create_radial(cx0, cy0, radius0, cx1,
      cy1, radius1);
  }}
  return result;
}
