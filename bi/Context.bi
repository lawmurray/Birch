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
    cairo_destroy(cr);
    }}
  }
  
  /*
   * Paths
   * -----
   */
  function newPath() {
    cpp{{
    cairo_new_path(cr);
    }}
  }
  
  function closePath() {
    cpp{{
    cairo_close_path(cr);
    }}
  }

  function arc(xc:Real, yc:Real, radius:Real, angle1:Real, angle2:Real) {
    cpp{{
    cairo_arc(cr, xc_, yc_, radius_, angle1_, angle2_);
    }}
  }

  function arcNegative(xc:Real, yc:Real, radius:Real, angle1:Real, angle2:Real) {
    cpp{{
    cairo_arc_negative(cr, xc_, yc_, radius_, angle1_, angle2_);
    }}
  }

  function curveTo(x1:Real, y1:Real, x2:Real, y2:Real, x3:Real, y3:Real) {
    cpp{{
    cairo_curve_to(cr, x1_, y1_, x2_, y2_, x3_, y3_);
    }}
  }

  function lineTo(x:Real, y:Real) {
    cpp{{
    cairo_line_to(cr, x_, y_);
    }}
  }

  function moveTo(x:Real, y:Real) {
    cpp{{
    cairo_move_to(cr, x_, y_);
    }}
  }

  function rectangle(x:Real, y:Real, width:Real, height:Real) {
    cpp{{
    cairo_rectangle(cr, x_, y_, width_, height_);
    }}
  }

  function relCurveTo(dx1:Real, dy1:Real, dx2:Real, dy2:Real, dx3:Real, dy3:Real) {
    cpp{{
    cairo_curve_to(cr, dx1_, dy1_, dx2_, dy2_, dx3_, dy3_);
    }}
  }

  function relLineTo(dx:Real, dy:Real) {
    cpp{{
    cairo_line_to(cr, dx_, dy_);
    }}
  }

  function relMoveTo(dx:Real, dy:Real) {
    cpp{{
    cairo_move_to(cr, dx_, dy_);
    }}
  }
  
  function stroke() {
    cpp{{
    cairo_stroke(cr);
    }}
  }

  function strokePreserve() {
    cpp{{
    cairo_stroke_preserve(cr);
    }}
  }

  function fill() {
    cpp{{
    cairo_fill(cr);
    }}
  }

  function fillPreserve() {
    cpp{{
    cairo_fill_preserve(cr);
    }}
  }

  function paint() {
    cpp{{
    cairo_paint(cr);
    }}
  }
  
  /*
   * Transformations
   * ---------------
   */
  function translate(tx:Real, ty:Real) {
    cpp{{
    cairo_translate(cr, tx_, ty_);
    }}
  }

  function scale(sx:Real, sy:Real) {
    cpp{{
    cairo_scale(cr, sx_, sy_);
    }}
  }
  
  function rotate(angle:Real) {
    cpp{{
    cairo_rotate(cr, angle_);
    }}
  }
  
  function deviceToUserDistance(ux:Real, uy:Real) -> (Real, Real) {
    ux1:Real <- ux;
    uy1:Real <- uy;
    cpp{{
    cairo_device_to_user_distance(cr, &ux1_, &uy1_);
    }}
    return (ux1, uy1);
  }
  
  /*
   * Sources
   * -------
   */
  function setSourceRGB(red:Real, green:Real, blue:Real) {
    cpp{{
    cairo_set_source_rgb(cr, red_, green_, blue_);
    }}
  }

  function setSourceRGBA(red:Real, green:Real, blue:Real, alpha:Real) {
    cpp{{
    cairo_set_source_rgba(cr, red_, green_, blue_, alpha_);
    }}
  }
  
  function setSource(pattern:Pattern) {
    cpp{{
    cairo_set_source(cr, pattern_->pattern);
    }}
  }
  
  function setLineWidth(width:Real) {
    cpp{{
    cairo_set_line_width(cr, width_);
    }}
  }
  
  /*
   * Groups
   * ------
   */
  function pushGroup() {
    cpp{{
    cairo_push_group(cr);
    }}
  }
  
  function popGroupToSource() {
    cpp{{
    cairo_pop_group_to_source(cr);
    }}
  }
}

function create(surface:Surface) -> Context {
  cr:Context;
  cpp{{
  cr_->cr = cairo_create(surface_->surface);
  }}
  return cr;
}
