/**
 * JSON file writer.
 */
class JSONWriter < MemoryWriter {
  hpp{{
  libubjpp::value top;
  }}

  /**
   * Path.
   */
  path:String;

  /**
   * Open file.
   *
   *   - path: Path.
   */
  function open(path:String) {
    this.path <- path;
    cpp{{
    group = &top;
    }}
  }

  function flush() {
    cpp{{
    if (!path_.empty() && group) {
      std::ofstream stream(path_);
      libubjpp::JSONGenerator generator(stream);
      generator.write(*group);
    }
    }}
  }
}
