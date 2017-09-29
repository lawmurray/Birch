/**
 * File input stream.
 */
class FileInputStream < InputStream {
  /**
   * Open file.
   *
   *   - path: Path.
   *   - mode: Mode. 
   */
  function open(path:String, mode:String) {
    file <- fopen(path, mode);
  }

  /**
   * Open file with default mode.
   *
   *   - path: Path.
   */
  function open(path:String) {
    open(path, "r");
  }

  /**
   * Close file.
   */
  function close() {
    assert file?;
    fclose(file!);
  }

  /**
   * Check for end-of-file.
   */
  function eof() -> Boolean {
    assert file?;
    cpp{{
    return ::feof(file_.get());
    }}
  }
}
