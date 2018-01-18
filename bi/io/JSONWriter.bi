/**
 * JSON file writer.
 */
class JSONWriter < MemoryWriter {
  hpp{{
  libubjpp::value top;
  }}

  /**
   * Save data to file.
   *
   *   - path: Path.
   */
  function save(path:String) {
    cpp{{
    std::ofstream stream(path_);
    libubjpp::JSONGenerator generator(stream);
    generator.write(top);
    }}
  }
}
