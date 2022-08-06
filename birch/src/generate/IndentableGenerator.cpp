/**
 * @file
 */
#include "src/generate/IndentableGenerator.hpp"

birch::IndentableGenerator::IndentableGenerator(std::ostream& base,
    const int level, const bool header) :
    base(base),
    level(level),
    header(header),
    indent(2 * level, ' ') {
  /* pre-condition */
  assert(level >= 0);
}

birch::IndentableGenerator& birch::IndentableGenerator::operator<<(const Name* o) {
  o->accept(this);
  return *this;
}

birch::IndentableGenerator& birch::IndentableGenerator::operator<<(const File* o) {
  o->accept(this);
  return *this;
}

birch::IndentableGenerator& birch::IndentableGenerator::operator<<(const Package* o) {
  o->accept(this);
  return *this;
}

birch::IndentableGenerator& birch::IndentableGenerator::operator<<(
    const Expression* o) {
  o->accept(this);
  return *this;
}

birch::IndentableGenerator& birch::IndentableGenerator::operator<<(
    const Statement* o) {
  o->accept(this);
  return *this;
}

birch::IndentableGenerator& birch::IndentableGenerator::operator<<(const Type* o) {
  o->accept(this);
  return *this;
}

birch::IndentableGenerator& birch::IndentableGenerator::operator<<(
    const Location* o) {
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

void birch::IndentableGenerator::in() {
  ++level;
  indent.append(2, ' ');
}

void birch::IndentableGenerator::out() {
  /* pre-condition */
  assert(level > 0);

  --level;
  indent.resize(2 * level);
}
