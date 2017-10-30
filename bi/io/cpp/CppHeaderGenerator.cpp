/**
 * @file
 */
#include "bi/io/cpp/CppHeaderGenerator.hpp"

#include "bi/io/cpp/CppBaseGenerator.hpp"
#include "bi/visitor/Gatherer.hpp"
#include "bi/primitive/poset.hpp"
#include "bi/primitive/definitely.hpp"

#include "boost/filesystem.hpp"

bi::CppHeaderGenerator::CppHeaderGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header) {
  //
}

void bi::CppHeaderGenerator::visit(const Package* o) {
  line("#include \"bi/libbirch.hpp\"");
  for (auto header : o->headers) {
    boost::filesystem::path include = header->path;
    include.replace_extension(".hpp");
    line("#include \"" << include.string() << "\"");
  }

  /* gather important objects */
  Gatherer<Class> classes;
  Gatherer<Alias> aliases;
  Gatherer<Raw> raws;
  Gatherer<GlobalVariable> globals;
  Gatherer<Function> functions;
  Gatherer<Fiber> fibers;
  Gatherer<Program> programs;
  Gatherer<BinaryOperator> binaries;
  Gatherer<UnaryOperator> unaries;
  for (auto source : o->sources) {
    source->accept(&classes);
    source->accept(&aliases);
    source->accept(&raws);
    source->accept(&globals);
    source->accept(&functions);
    source->accept(&fibers);
    source->accept(&programs);
    source->accept(&binaries);
    source->accept(&unaries);
  }

  /* raw C++ code for headers */
  for (auto o1 : raws) {
    *this << o1;
  }
  line("");
  line("namespace bi {");

  /* forward class declarations */
  for (auto o : classes) {
    if (!o->braces->isEmpty()) {
      genTemplateParams(o);
      line("class " << o->name << ';');
    }
  }
  line("");

  /* typedef declarations */
  for (auto o : aliases) {
    line("using " << o->name << " = " << o->base << ';');
  }
  line("");

  /* forward super type declarations */
  for (auto o : classes) {
    if (!o->braces->isEmpty()) {
      if (o->isGeneric()) {
        genTemplateParams(o);
      } else {
        start("template<>");
      }
      middle(" struct super_type<" << o->name);
      genTemplateArgs(o);
      finish("> {");
      in();
      start("typedef ");
      if (!o->base->isEmpty()) {
        auto super = dynamic_cast<const ClassType*>(o->base);
        assert(super);
        middle(super->name);
        if (!super->typeArgs->isEmpty()) {
          middle('<' << super->typeArgs << '>');
        }
      } else {
        middle("Object_");
      }
      finish(" type;");
      out();
      line("};");
    }
  }

  /* class definitions; even with the forward declarations above, base
   * classes must be defined before their derived classes, so these are
   * gathered and sorted first */
  poset<Type*,definitely> sorted;
  for (auto o : classes) {
    sorted.insert(new ClassType(o));
  }
  for (auto iter = sorted.rbegin(); iter != sorted.rend(); ++iter) {
    *this << (*iter)->getClass();
  }

  /* global variables */
  for (auto o : globals) {
    *this << o;
  }
  line("");

  /* functions and fibers */
  line("namespace func {");
  for (auto o : functions) {
    *this << o;
  }
  for (auto o : fibers) {
    *this << o;
  }
  line("}\n");

  /* programs */
  for (auto o : programs) {
    *this << o;
  }
  line("");
  line("}\n");

  /* operators */
  for (auto o : binaries) {
    *this << o;
  }
  for (auto o : unaries) {
    *this << o;
  }
}
