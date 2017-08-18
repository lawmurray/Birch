import basic;
import io.OutputStream;

/**
 * File output stream.
 */
class FileOutputStream(file:String) < OutputStream(open(file, "w")) {
  //
}
