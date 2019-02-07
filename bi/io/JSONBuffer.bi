cpp{{
#include "boost/filesystem.hpp"

#include <fstream>
}}

/**
 * JSON file buffer.
 */
class JSONBuffer < MemoryBuffer {  
  /**
   * Load into memory buffer from a JSON file.
   *
   * - path: Path of the file.
   */
  function load(path:String) {
    success:Boolean <- false;
    cpp{{
    std::ifstream stream(path_);
    if (stream.is_open()) {
      libubjpp::JSONDriver driver;
      if (auto result = driver.parse(stream)) {
        self->root = result.get();
        success_ = true;
      }
    }
    }}
    if (!success) {
      stderr.print("warning: could not load from \'" + path + "'\n");
    }
  }

  /**
   * Save memory buffer to a JSON file.
   *
   * - path: Path of the file.
   */
  function save(path:String) {
    mkdir(path);
    success:Boolean <- false;
    cpp{{
    std::ofstream stream(path_);
    if (stream.is_open()) {
      libubjpp::JSONGenerator generator(stream);
      generator.apply(self->root);
      success_ = true;
    }
    }}
    if (!success) {
      stderr.print("warning: could not save to \'" + path + "'\n");
    }
  }
}
