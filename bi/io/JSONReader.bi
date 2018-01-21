/**
 * JSON file reader.
 */
class JSONReader < MemoryReader {
  hpp{{
  libubjpp::value top;
  }}

  /**
   * Open file.
   *
   *   - path: Path.
   */
  function open(path:String) {
    cpp{{
    libubjpp::JSONDriver driver;
    std::ifstream stream(path_);
    auto result = driver.parse(stream);
    if (result) {
      top = result.get();
      group = &top;
    }
    }}
  }
}
