/**
 * @file
 */
#include "src/generate/CppStructGenerator.hpp"

#include "src/visitor/Gatherer.hpp"
#include "src/primitive/string.hpp"

birch::CppStructGenerator::CppStructGenerator(std::ostream& base,
    const int level, const bool header, const bool includeInline,
    const bool includeLines, const Struct* currentStruct) :
    CppGenerator(base, level, header, includeInline, includeLines),
    currentStruct(currentStruct) {
  //
}

void birch::CppStructGenerator::visit(const Struct* o) {
  if (!o->isAlias() && !o->braces->isEmpty()) {
    Gatherer<MemberVariable> memberVariables;
    Gatherer<MemberPhantom> memberPhantoms;
    o->accept(&memberVariables);
    o->accept(&memberPhantoms);

    if (header) {
      genDoc(o->loc);
      genTemplateParams(o);
      genSourceLine(o->loc);
      start("struct " << o->name);
      if (!o->base->isEmpty()) {
        middle(" : public ");
        genBase(o);
      }
      finish(" {");
      line("public:");
      in();

      /* generic types */
      for (auto typeParam : *o->typeParams) {
        genSourceLine(o->loc);
        line("using " << typeParam << "_ = " << typeParam << ';');
      }

      /* boilerplate */
      genSourceLine(o->loc);
      start("MEMBIRCH_STRUCT(" << o->name);
      if (!o->base->isEmpty()) {
        middle(", ");
        genBase(o);
      }
      finish(')');

      genSourceLine(o->loc);
      start("MEMBIRCH_STRUCT_MEMBERS(");
      if (memberVariables.size() + memberPhantoms.size() > 0) {
        bool first = true;
        for (auto o : memberVariables) {
          if (!first) {
            middle(", ");
          }
          first = false;
          middle(o->name);
        }
        for (auto o : memberPhantoms) {
          if (!first) {
            middle(", ");
          }
          first = false;
          middle(o->name);
        }
      }
      finish(')');

      /* dereference and member access through pointer operators; these
       * allow struct objects to be dereferenced with unary `*` (an identity
       * operation) and members accessed through binary `->` (acting as if
       * `.`), simplifying code generation */
      line("auto operator->() {");
      in();
      line("return this;");
      out();
      line("}\n");

      line("const auto operator->() const {");
      in();
      line("return this;");
      out();
      line("}\n");
    }

    /* constructor */
    if (!header) {
      genTemplateParams(o);
      genSourceLine(o->loc);
      start(o->name);
      genTemplateArgs(o);
      middle("::");
    } else {
      genSourceLine(o->loc);
      start("explicit ");
    }
    middle(o->name);
    middle('(' << o->params << ')');
    if (header) {
      finish(";\n");
    } else {
      ++inConstructor;
      bool first = true;
      if (!o->base->isEmpty()) {
        finish(" :");
        first = false;
        in();
        in();
        genSourceLine(o->loc);
        start("base_type_(" << o->args << ')');
      }
      for (auto o : memberVariables) {
        if (first) {
          finish(" :");
          first = false;
          in();
          in();
        } else {
          finish(',');
        }
        genSourceLine(o->loc);
        start(o->name << '(');
        genInit(o);
        middle(')');
      }
      --inConstructor;
      if (!first) {
        out();
        out();
      }
      finish(" {");
      in();
      line("//");
      out();
      line("}\n");
    }

    /* member variables */
    *this << o->braces->strip();

    /* end struct */
    if (header) {
      out();
      line("};\n");
    }
  }
}

void birch::CppStructGenerator::visit(const MemberVariable* o) {
  if (header) {
    genDoc(o->loc);
    line(o->type << ' ' << o->name << ';');
  }
}

void birch::CppStructGenerator::genBase(const Struct* o) {
  auto base = dynamic_cast<const NamedType*>(o->base);
  if (base) {
    middle(base->name);
    if (!base->typeArgs->isEmpty()) {
      middle('<' << base->typeArgs << '>');
    }
  }
}
