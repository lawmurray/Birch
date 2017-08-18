import basic;
import io.OutputStream;

/**
 * File output stream.
 */
class FileOutputStream(file:String) < OutputStream(open(file, "w")) {
  /**
   * Close the file.
   */
  function close() {
    //close(stream);
  }
}
