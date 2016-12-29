/**
 * @file
 */
#include "bi/io/indentable_ostream.hpp"

bi::indentable_ostream::indentable_ostream(std::ostream& base,
    const int level, const bool header) :
    base(base), level(level), header(header), indent(2*level, ' ') {
  /* pre-condition */
  assert(level >= 0);
}

bi::indentable_ostream& bi::indentable_ostream::operator<<(
    const bi::Location* o) {
  if (o->file) {
    /* the format here matches that of g++ and clang++ such that Eclipse,
     * when parsing the error output, is able to annotate lines within the
     * editor */
    *this << o->file->path;
    /*if (o->lastLine > o->firstLine) {
      *this << ":" << o->firstLine << '-' << o->lastLine;
    } else {*/
      *this << ':' << o->firstLine;
      /*if (o->lastCol > o->firstCol) {
        *this << ":" << o->firstCol << '-' << o->lastCol;
      } else {*/
        *this << ':' << o->firstCol;
      //}
    //}
    *this << ": ";
  }
  return *this;
}
