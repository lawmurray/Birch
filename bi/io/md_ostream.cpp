/**
 * @file
 */
#include "bi/io/md_ostream.hpp"

#include "bi/primitive/encode.hpp"

bi::md_ostream::md_ostream(std::ostream& base) :
    bih_ostream(base) {
  //
}

void bi::md_ostream::visit(const GlobalVariable* o) {
  if (!o->loc->doc.empty()) {
    genDoc(o->loc);
    in(); in();
    bih_ostream::visit(o);
    out(); out();
    *this << "\n\n";
  }
}

void bi::md_ostream::visit(const MemberVariable* o) {
  if (!o->loc->doc.empty()) {
    genDoc(o->loc);
    in(); in();
    bih_ostream::visit(o);
    out(); out();
    *this << "\n\n";
  }
}

void bi::md_ostream::visit(const Function* o) {
  if (!o->loc->doc.empty()) {
    genDoc(o->loc);
    in(); in();
    bih_ostream::visit(o);
    out(); out();
    *this << "\n\n";
  }
}

void bi::md_ostream::visit(const Fiber* o) {
  if (!o->loc->doc.empty()) {
    genDoc(o->loc);
    in(); in();
    bih_ostream::visit(o);
    out(); out();
    *this << "\n\n";
  }
}

void bi::md_ostream::visit(const Program* o) {
  if (!o->loc->doc.empty()) {
    genDoc(o->loc);
    in(); in();
    bih_ostream::visit(o);
    out(); out();
    *this << "\n\n";
  }
}

void bi::md_ostream::visit(const MemberFunction* o) {
  if (!o->loc->doc.empty()) {
    genDoc(o->loc);
    in(); in();
    bih_ostream::visit(o);
    out(); out();
    *this << "\n\n";
  }
}

void bi::md_ostream::visit(const MemberFiber* o) {
  if (!o->loc->doc.empty()) {
    genDoc(o->loc);
    in(); in();
    bih_ostream::visit(o);
    out(); out();
    *this << "\n\n";
  }
}

void bi::md_ostream::visit(const BinaryOperator* o) {
  if (!o->loc->doc.empty()) {
    genDoc(o->loc);
    in(); in();
    bih_ostream::visit(o);
    out(); out();
    *this << "\n\n";
  }
}

void bi::md_ostream::visit(const UnaryOperator* o) {
  if (!o->loc->doc.empty()) {
    genDoc(o->loc);
    in(); in();
    bih_ostream::visit(o);
    out(); out();
    *this << "\n\n";
  }
}

void bi::md_ostream::visit(const AssignmentOperator* o) {
  if (!o->loc->doc.empty()) {
    genDoc(o->loc);
    in(); in();
    bih_ostream::visit(o);
    out(); out();
    *this << "\n\n";
  }
}

void bi::md_ostream::visit(const ConversionOperator* o) {
  if (!o->loc->doc.empty()) {
    genDoc(o->loc);
    in(); in();
    bih_ostream::visit(o);
    out(); out();
    *this << "\n\n";
  }
}

void bi::md_ostream::visit(const Class* o) {
  if (!o->loc->doc.empty()) {
    genDoc(o->loc);
    in(); in();
    bih_ostream::visit(o);
    out(); out();
    *this << "\n\n";
  }
}

void bi::md_ostream::visit(const Alias* o) {
  if (!o->loc->doc.empty()) {
    genDoc(o->loc);
    in(); in();
    bih_ostream::visit(o);
    out(); out();
    *this << "\n\n";
  }
}

void bi::md_ostream::visit(const Basic* o) {
  if (!o->loc->doc.empty()) {
    genDoc(o->loc);
    in(); in();
    bih_ostream::visit(o);
    out(); out();
    *this << "\n\n";
  }
}

void bi::md_ostream::visit(const Import* o) {
  //
}

void bi::md_ostream::genDoc(const Location* o) {
  *this << comment(o->doc) << "\n";
}
