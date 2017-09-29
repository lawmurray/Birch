/**
 * File output stream.
 */
class FileOutputStream < OutputStream {
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
    open(path, "w");
  }

  /**
   * Close the file.
   */
  function close() {
    assert file?;
    fclose(file!);
  }
}
