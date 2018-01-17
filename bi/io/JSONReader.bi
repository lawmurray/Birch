/**
 * JSON file reader.
 */
class JSONReader < MemoryReader {
  hpp{{
  libubjpp::value top;
  }}

  /**
   * Load data from file.
   *
   *   - path: Path.
   */
  function load(path:String) {
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
