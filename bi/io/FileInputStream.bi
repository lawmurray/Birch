/**
 * File input stream.
 */
class FileInputStream(file:String) < InputStream(fopen(file, "r")) {
  /**
   * Close the file.
   */
  function close() {
    fclose(stream);
  }
}
