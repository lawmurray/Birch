/**
 * @file
 */
#include "bi/io/cpp/CppClassGenerator.hpp"

#include "bi/io/cpp/CppConstructorGenerator.hpp"
#include "bi/io/cpp/CppMemberCoroutineGenerator.hpp"
#include "bi/primitive/encode.hpp"

bi::CppClassGenerator::CppClassGenerator(std::ostream& base, const int level,
    const bool header) :
    CppBaseGenerator(base, level, header),
    type(nullptr) {
  //
}

void bi::CppClassGenerator::visit(const Class* o) {
  type = o;

  /* start boilerplate */
  if (header) {
    start("class " << o->name << " : public ");
    if (!o->base->isEmpty()) {
      auto type = dynamic_cast<const ClassType*>(o->base);
      assert(type);
      middle(type->name);
    } else {
      middle("Object");
    }
    finish(" {");
    line("public:");
    in();
    line("typedef " << o->name << " this_type;");
    if (o->base->isEmpty()) {
      line("typedef Object super_type;");
    } else {
      auto type = dynamic_cast<const ClassType*>(o->base);
      assert(type);
      line("typedef " << type->name << " super_type;");
    }
    line("");
  }

  /* constructor */
  CppConstructorGenerator auxConstructor(base, level, header);
  auxConstructor << o;

  /* destructor */
  if (header) {
    line("virtual ~" << o->name << "() {");
    in();
    line("//");
    out();
    line("}\n");
  }

  /* clone function */
  if (!header) {
    start("bi::type::");
  } else {
    start("virtual ");
  }
  middle(o->name << "* ");
  if (!header) {
    middle("bi::type::" << o->name << "::");
  }
  middle("clone()");
  if (header) {
    finish(";\n");
  } else {
    finish(" {");
    in();
    line("return copy_object(this);");
    out();
    line("}\n");
  }

  /* member parameters */
  for (auto iter = o->parens->begin(); iter != o->parens->end(); ++iter) {
    *this << *iter;
  }

  /* member variables and functions */
  *this << o->braces;

  /* end boilerplate */
  if (header) {
    out();
    line("};\n");
  }
}

void bi::CppClassGenerator::visit(const MemberParameter* o) {
  if (header) {
    line(o->type << ' ' << o->name << ';');
  }
}

void bi::CppClassGenerator::visit(const MemberVariable* o) {
  if (header) {
    line(o->type << ' ' << o->name << ';');
  }
}

void bi::CppClassGenerator::visit(const MemberFunction* o) {
  if (header) {
    start("virtual ");
  } else {
    start("");
  }
  middle(o->returnType << ' ');
  if (!header) {
    middle("bi::type::" << type->name << "::");
  }
  middle(internalise(o->name->str()) << o->parens);
  //middle(" const");
  if (header) {
    finish(';');
  } else {
    finish(" {");
    in();

    /* body */
    CppBaseGenerator auxBase(base, level, header);
    auxBase << o->braces;

    out();
    finish("}\n");
  }
}

void bi::CppClassGenerator::visit(const MemberCoroutine* o) {
  CppMemberCoroutineGenerator auxMemberCoroutine(type, base, level, header);
  auxMemberCoroutine << o;
}

void bi::CppClassGenerator::visit(const AssignmentOperator* o) {
  if (!o->braces->isEmpty()) {
    start("");
    if (!header) {
      middle("bi::type::");
    }
    start(type->name << "& ");
    if (!header) {
      middle("bi::type::" << type->name << "::");
    }
    middle("operator=(" << o->single << ')');
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      CppBaseGenerator auxBase(base, level, header);
      auxBase << o->braces;
      line("return *this;");
      out();
      finish("}\n");
    }
  }
}

void bi::CppClassGenerator::visit(const ConversionOperator* o) {
  if (!o->braces->isEmpty()) {
    if (!header) {
      start("bi::type::" << type->name << "::");
    } else {
      start("");
    }
    middle("operator " << o->returnType << "()");
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      CppBaseGenerator auxBase(base, level, header);
      auxBase << o->braces;
      out();
      finish("}\n");
    }
  }
}
