cpp{{
#include "libubjpp/libubjpp.hpp"
#include "libubjpp/json/JSONGenerator.hpp"
#include "boost/filesystem.hpp"

#include <fstream>
}}

/**
 * JSON file writer.
 */
class JSONWriter < MemoryWriter {
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
   * Save data to file.
   */
  function save() {
    success:Boolean <- false;
    cpp{{
    boost::filesystem::path path = path_;
    if (!path.parent_path().empty()) {
      boost::filesystem::create_directories(path.parent_path());
    }
    std::ofstream stream(path_);
    if (stream.is_open()) {
      libubjpp::JSONGenerator generator(stream);
      generator.apply(top);
      success_ = true;
    }
    }}
    if (!success) {
      stderr.print("warning: could not save to \'" + path + "'\n");
    }
  }
}

/**
 * Factory function.
 */
function JSONWriter(path:String) -> JSONWriter {
  writer:JSONWriter;
  writer.make(path);
  return writer;
}
