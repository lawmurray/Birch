/**
 * Structured stream.
 *
 * When reading and writing objects, defers to the `read()` and `write()`
 * member functions, declared in `Object`.
 */
class StructuredStream {
  /**
   * Root value.
   */
  root:Value?;

  /**
   * Load from a file.
   *
   * - path: Path of the file.
   */
  function load(path:String) {
    parser:JSONParser;
    root <- parser.parse(path);
    if !root {
      stderr.print("warning: could not load from \'" + path + "'\n");
    }
  }

  /**
   * Save to a file.
   *
   * - path: Path of the file.
   */
  function save(path:String) {
    mkdir(path);
    generator:JSONGenerator;
    auto success <- generator.generate(path, root);
    if !success {
      stderr.print("warning: could not save to \'" + path + "'\n");
    }
  }

  /**
   * Set the root node as an object.
   */
  function setObject() {
    root:ObjectValue;
    this.root <- root;
  }
  
  /**
   * Set the root node as an array.
   */
  function setArray() {
    root:ArrayValue;
    this.root <- root;
  }
}
