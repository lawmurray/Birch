import basic;
import io.OutputStream;

/**
 * File output stream.
 */
class FileOutputStream(file:String) < OutputStream(fopen(file, "w")) {
  /**
   * Close the file.
   */
  function close() {
    fclose(stream);
  }
}
