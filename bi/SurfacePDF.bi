hpp{{
#include <cairo/cairo-pdf.h>
}}

/**
 * A PDF surface. This extends the usual Cairo interface.
 */
class SurfacePDF < Surface {

}

function createPDF(filename:String, widthInPoints:Real, heightInPoints:Real)
    -> Surface {
  surface:SurfacePDF;
  cpp{{
  surface_->surface = cairo_pdf_surface_create(filename_.c_str(),
      widthInPoints_, heightInPoints_);
  }}
  return surface;
}