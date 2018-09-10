/**
 * Object that can be serialized using Reader and Writer.
 */
class Serializable {
  /**
   * Read.
   */
  function read(reader:Reader) {
    //
  }
  
  /**
   * Write.
   */
  function write(writer:Writer) {
    //
  }

  /**
   * Read.
   */
  function input(reader:Reader) {
    stderr.print("input(reader:Reader) is deprecated, use read(reader:Reader)\n");
    read(reader);
  }
  
  /**
   * Write.
   */
  function output(writer:Writer) {
    stderr.print("output(writer:Writer) is deprecated, use write(writer:Writer)\n");
    write(writer);
  }
}
