/**
 * JSON file reader.
 */
class JSONReader < MemoryReader {
  hpp{{
  libubjpp::value top;
  }}
  
  /**
   * File path.
   */
  path:String;
  
  /**
   * Constructor.
   */
  function make(path:String) {
    this.path <- path;
    cpp{{
    group = &top;
    }}
  }
  
  /**
   * Load data from file.
   */
  function load() {
    success:Boolean;
    cpp{{
    std::ifstream stream(path_);
    success_ = stream.is_open();
    }}
    if (success) {
      cpp{{
      libubjpp::JSONDriver driver;
      if (auto result = driver.parse(stream)) {
        top = result.get();
        group = &top;
      }
      }}
    } else {
      stderr.print("warning: could not load from \'" + path + "'\n");
    }
  }
}

/**
 * Factory function.
 */
function JSONReader(path:String) -> JSONReader {
  reader:JSONReader;
  reader.make(path);
  return reader;
}
