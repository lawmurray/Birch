/**
 * Cairo surface.
 */
class Surface {
  hpp{{
  cairo_surface_t* surface;
  }}
  
  function destroy() {
    cpp{{
    cairo_surface_destroy(self->surface);
    }}
  }
}
