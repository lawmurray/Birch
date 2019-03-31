/**
 * A PNG surface. This extends the usual Cairo interface.
 */
class SurfacePNG < Surface {
  filename:String;
  
  function destroy() {
    /* write the file on destruction */
    mkdir(filename);
    cpp{{
    cairo_surface_write_to_png(self->surface, self->filename.c_str());
    }}    
    super.destroy();
  }
}

function createPNG(filename:String, width:Integer, height:Integer)
    -> Surface {
  surface:SurfacePNG;
  cpp{{
  surface->filename = filename;
  surface->surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32,
      width, height);
  }}
  return surface;
}
