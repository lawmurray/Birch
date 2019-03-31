hpp{{
#include <cairo/cairo.h>
}}

/**
 * Cairo graphics context.
 */
class Context {
  hpp{{
  cairo_t* cr;
  }}

  function destroy() {
    cpp{{
    cairo_destroy(self->cr);
    }}
  }
  
  /*
   * Paths
   * -----
   */
  function newPath() {
    cpp{{
    cairo_new_path(self->cr);
    }}
  }
  
  function closePath() {
    cpp{{
    cairo_close_path(self->cr);
    }}
  }

  function arc(xc:Real, yc:Real, radius:Real, angle1:Real, angle2:Real) {
    cpp{{
    cairo_arc(self->cr, xc, yc, radius, angle1, angle2);
    }}
  }

  function arcNegative(xc:Real, yc:Real, radius:Real, angle1:Real, angle2:Real) {
    cpp{{
    cairo_arc_negative(self->cr, xc, yc, radius, angle1, angle2);
    }}
  }

  function curveTo(x1:Real, y1:Real, x2:Real, y2:Real, x3:Real, y3:Real) {
    cpp{{
    cairo_curve_to(self->cr, x1, y1, x2, y2, x3, y3);
    }}
  }

  function lineTo(x:Real, y:Real) {
    cpp{{
    cairo_line_to(self->cr, x, y);
    }}
  }

  function moveTo(x:Real, y:Real) {
    cpp{{
    cairo_move_to(self->cr, x, y);
    }}
  }

  function rectangle(x:Real, y:Real, width:Real, height:Real) {
    cpp{{
    cairo_rectangle(self->cr, x, y, width, height);
    }}
  }

  function relCurveTo(dx1:Real, dy1:Real, dx2:Real, dy2:Real, dx3:Real, dy3:Real) {
    cpp{{
    cairo_curve_to(self->cr, dx1, dy1, dx2, dy2, dx3, dy3);
    }}
  }

  function relLineTo(dx:Real, dy:Real) {
    cpp{{
    cairo_line_to(self->cr, dx, dy);
    }}
  }

  function relMoveTo(dx:Real, dy:Real) {
    cpp{{
    cairo_move_to(self->cr, dx, dy);
    }}
  }
  
  function stroke() {
    cpp{{
    cairo_stroke(self->cr);
    }}
  }

  function strokePreserve() {
    cpp{{
    cairo_stroke_preserve(self->cr);
    }}
  }

  function fill() {
    cpp{{
    cairo_fill(self->cr);
    }}
  }

  function fillPreserve() {
    cpp{{
    cairo_fill_preserve(self->cr);
    }}
  }

  function paint() {
    cpp{{
    cairo_paint(self->cr);
    }}
  }
  
  /*
   * Transformations
   * ---------------
   */
  function translate(tx:Real, ty:Real) {
    cpp{{
    cairo_translate(self->cr, tx, ty);
    }}
  }

  function scale(sx:Real, sy:Real) {
    cpp{{
    cairo_scale(self->cr, sx, sy);
    }}
  }
  
  function rotate(angle:Real) {
    cpp{{
    cairo_rotate(self->cr, angle);
    }}
  }
  
  function deviceToUserDistance(ux:Real, uy:Real) -> (Real, Real) {
    ux1:Real <- ux;
    uy1:Real <- uy;
    cpp{{
    cairo_device_to_user_distance(self->cr, &ux1, &uy1);
    }}
    return (ux1, uy1);
  }
  
  /*
   * Sources
   * -------
   */
  function setSourceRGB(red:Real, green:Real, blue:Real) {
    cpp{{
    cairo_set_source_rgb(self->cr, red, green, blue);
    }}
  }

  function setSourceRGBA(red:Real, green:Real, blue:Real, alpha:Real) {
    cpp{{
    cairo_set_source_rgba(self->cr, red, green, blue, alpha);
    }}
  }
  
  function setSource(pattern:Pattern) {
    cpp{{
    cairo_set_source(self->cr, pattern->pattern);
    }}
  }
  
  function setLineWidth(width:Real) {
    cpp{{
    cairo_set_line_width(self->cr, width);
    }}
  }

  /*
   * Text
   * ----
   */
  function showText(utf8:String) {
    cpp{{
    cairo_show_text(self->cr, utf8.c_str());
    }}
  }

  function setFontSize(size:Real) {
    cpp{{
    cairo_set_font_size(self->cr, size);
    }}
  }
  
  /*
   * Groups
   * ------
   */
  function pushGroup() {
    cpp{{
    cairo_push_group(self->cr);
    }}
  }
  
  function popGroupToSource() {
    cpp{{
    cairo_pop_group_to_source(self->cr);
    }}
  }
}

function create(surface:Surface) -> Context {
  cr:Context;
  cpp{{
  cr->cr = cairo_create(surface->surface);
  }}
  return cr;
}
