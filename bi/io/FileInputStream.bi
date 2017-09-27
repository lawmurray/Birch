/**
 * File input stream.
 */
class FileInputStream(file:String) < InputStream(fopen(file, "r")) {
  /**
   * Check for end-of-file.
   */
  function eof() -> Boolean {
    cpp{{
    return ::feof(stream_);
    }}
  }

  /**
   * Close the file.
   */
  function close() {
    fclose(stream);
  }
}
