hpp{{
#include <cairo/cairo-svg.h>
}}

/**
 * An SVG surface. This extends the usual Cairo interface.
 */
class SurfaceSVG < Surface {

}

function createSVG(filename:String, widthInPoints:Real, heightInPoints:Real)
      -> Surface {
  mkdir(filename);
  surface:SurfaceSVG;
  cpp{{
  surface_->surface = cairo_svg_surface_create(filename_.c_str(),
      widthInPoints_, heightInPoints_);
  }}
  return surface;
}
