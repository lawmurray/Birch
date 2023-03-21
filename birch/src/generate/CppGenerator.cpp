/**
 * @file
 */
#include "src/generate/CppGenerator.hpp"

#include "src/generate/CppClassGenerator.hpp"
#include "src/generate/CppStructGenerator.hpp"
#include "src/primitive/string.hpp"

birch::CppGenerator::CppGenerator(std::ostream& base, const int level,
    const bool header, const bool includeInline, const bool includeLines) :
    IndentableGenerator(base, level),
    header(header),
    includeInline(includeInline),
    includeLines(includeLines),
    inAssign(0),
    inGlobal(0),
    inConstructor(0),
    inOperator(0),
    inLambda(0),
    inMember(0),
    inSequence(0),
    inReturn(0) {
  //
}

void birch::CppGenerator::visit(const Name* o) {
  middle(sanitize(o->str()));
}

void birch::CppGenerator::visit(const ExpressionList* o) {
  middle(o->head);
  if (o->tail) {
    middle(", " << o->tail);
  }
}

void birch::CppGenerator::visit(const Literal<bool>* o) {
  middle(o->str);
}

void birch::CppGenerator::visit(const Literal<int64_t>* o) {
  middle(o->str);
}

void birch::CppGenerator::visit(const Literal<double>* o) {
  if (o->str == "nan") {
    middle("std::numeric_limits<Real>::quiet_NaN()");
  } else if (o->str == "inf") {
    middle("std::numeric_limits<Real>::infinity()");
  } else {
    middle("Real(" << o->str << ')');
  }
}

void birch::CppGenerator::visit(const Literal<const char*>* o) {
  middle("String(" << o->str << ')');
}

void birch::CppGenerator::visit(const Parentheses* o) {
  auto stripped = o->strip();
  if (stripped->isTuple()) {
    if (inAssign) {
      middle("std::tie");
    } else {
      middle("std::make_tuple");
    }
  }
  middle('(' << stripped << ')');
}

void birch::CppGenerator::visit(const Sequence* o) {
  if (o->single->isEmpty()) {
    middle("std::nullopt");
  } else if (!inSequence) {
    middle("numbirch::Array(");
    ++inSequence;
    middle("{ " << o->single << " }");
    --inSequence;
    middle(')');
  } else {
    middle("{ " << o->single << " }");
  }
}

void birch::CppGenerator::visit(const Cast* o) {
  middle("birch::optional_cast<" << o->returnType << ">(" << o->single << ')');
}

void birch::CppGenerator::visit(const Call* o) {
  middle(o->single << '(' << o->args << ')');
}

void birch::CppGenerator::visit(const BinaryCall* o) {
  if (isTranslatable(o->name->str())) {
    middle(o->left << ' ' << o->name->str() << ' ' << o->right);
  } else {
    middle(o->name << '(' << o->left << ", " << o->right << ')');
  }
}

void birch::CppGenerator::visit(const UnaryCall* o) {
  if (isTranslatable(o->name->str())) {
    /* always include the parentheses here, consider that in Birch, ++x is the
     * equivalent of +(+(x)), whereas in C++ it means increment */
    middle(o->name->str() << '(' << o->single << ')');
  } else {
    middle(o->name << '(' << o->single << ')');
  }
}

void birch::CppGenerator::visit(const Assign* o) {
  ++inAssign;
  if (*o->name == "<-?") {
    middle("birch::optional_assign(" << o->left << ", ");
    --inAssign;
    middle(o->right << ')');
  } else if (*o->name == "~") {
    middle(o->left << " = ");
    --inAssign;
    middle('(' << o->right << ")->random()");
  } else if (*o->name == "<~") {
    middle(o->left << " = ");
    --inAssign;
    middle('(' << o->right << ")->value()");
  } else if (*o->name == "~>") {
    middle('(' << o->right << ")->assign");
    --inAssign;
    middle('(' << o->left << ')');
  } else {
    middle(o->left << " = ");
    --inAssign;
    middle(o->right);
  }
}

void birch::CppGenerator::visit(const Slice* o) {
  middle(o->single << '(' << o->brackets << ')');
}

void birch::CppGenerator::visit(const Query* o) {
  middle("(bool(" << o->single << "))");
}

void birch::CppGenerator::visit(const Get* o) {
  middle("(*(" << o->single << "))");
}

void birch::CppGenerator::visit(const LambdaFunction* o) {
  middle("[=](" << o->params << ')');
  if (!o->returnType->isEmpty() && !o->returnType->isDeduced()) {
    middle(" -> " << o->returnType);
  }
  finish(" {");
  in();
  ++inLambda;
  *this << o->braces->strip();
  --inLambda;
  out();
  start("}");
}

void birch::CppGenerator::visit(const Span* o) {
  if (o->single->isEmpty()) {
    middle('0');
  } else {
    middle(o->single);
  }
}

void birch::CppGenerator::visit(const Range* o) {
  middle("std::make_pair(" << o->left << ", " << o->right << ')');
}

void birch::CppGenerator::visit(const Member* o) {
  if (o->left->isThis()) {
    middle("this->");
  } else if (o->left->isSuper()) {
    middle("this->base_type_::");
  } else {
    middle(o->left << "->");
  }
  ++inMember;
  middle(o->right);
  --inMember;
}

void birch::CppGenerator::visit(const This* o) {
  middle("membirch::Shared<this_type_>(this)");
}

void birch::CppGenerator::visit(const Super* o) {
  middle("membirch::Shared<base_type_>(this)");
}

void birch::CppGenerator::visit(const Global* o) {
  middle("birch::" << o->single);
}

void birch::CppGenerator::visit(const Nil* o) {
  middle("std::nullopt");
}

void birch::CppGenerator::visit(const Parameter* o) {
  middle("const " << o->type << "& " << o->name);
  if (!o->value->isEmpty()) {
    middle(" = " << o->value);
  }
}

void birch::CppGenerator::visit(const NamedExpression* o) {
  middle(o->name);
  if (!o->typeArgs->isEmpty()) {
    middle('<' << o->typeArgs << '>');
  }
}

void birch::CppGenerator::visit(const File* o) {
  /* raw C++ code */
  for (auto o1 : *o->root) {
    if (dynamic_cast<const Raw*>(o1)) {
      *this << o1;
    }
  }

  /* everything else */
  line("namespace birch {");
  for (auto o1 : *o->root) {
    if (!dynamic_cast<const Raw*>(o1)) {
      *this << o1;
    }
  }
  line('}');
}

void birch::CppGenerator::visit(const GlobalVariable* o) {
  ++inGlobal;
  genDoc(o->loc);
  genSourceLine(o->loc);
  start("");
  if (header) {
    middle("extern ");
  }
  start(o->type << ' ' << o->name);
  if (!header) {
    genInit(o);
  }
  finish(';');
  --inGlobal;
}

void birch::CppGenerator::visit(const LocalVariable* o) {
  if (o->has(LET)) {
    start("auto " << o->name);
  } else {
    start(o->type << ' ' << o->name);
  }
  genInit(o);
  finish(';');
}

void birch::CppGenerator::visit(const TupleVariable* o) {
  Name* tmp = new Name();
  int i = 0;
  line("auto " << tmp << " = " << o->value << ';');
  for (auto iter = o->locals->begin(); iter != o->locals->end(); ++iter) {
    auto local = dynamic_cast<const LocalVariable*>(*iter);
    assert(local);
    line("auto " << local->name << " = std::move(std::get<" << i++ <<
        ">(" << tmp << "));");
  }
}

void birch::CppGenerator::visit(const Function* o) {
  if ((includeInline || !o->isGeneric()) && !o->braces->isEmpty()) {
    genDoc(o->loc);
    genTemplateParams(o);
    genSourceLine(o->loc);
    start("");
    if (!header && o->isGeneric()) {
      middle("inline ");
    }
    middle(o->returnType << ' ' << o->name << '(' << o->params << ')');
    if (header) {
      finish(";");
    } else {
      finish(" {");
      in();
      *this << o->braces->strip();
      out();
      line("}\n");
    }
  }
}

void birch::CppGenerator::visit(const Program* o) {
  if (!o->braces->isEmpty()) {
    genDoc(o->loc);
    genSourceLine(o->loc);
    start("int " << o->name << "(int argc_, char** argv_)");
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();

      /* program options */
      if (o->params->width() > 0) {
        /* option variables */
        for (auto iter = o->params->begin(); iter != o->params->end(); ++iter) {
          auto param = dynamic_cast<const Parameter*>(*iter);
          assert(param);
          genSourceLine(o->loc);
          start("[[maybe_unused]] " << param->type << ' ' << param->name);
          if (!param->value->isEmpty()) {
            middle(" = " << param->value);
          }
          finish(';');
        }
        line("");

        /* option flags */
        line("enum {");
        in();
        for (auto param : *o->params) {
          auto name = dynamic_cast<const Parameter*>(param)->name;
          auto flag = sanitize(name->str()) + "FLAG_";
          line(flag << ',');
        }
        out();
        line("};");

        /* long options */
        genSourceLine(o->loc);
        line("int c_, option_index_;");
        genSourceLine(o->loc);
        line("option long_options_[] = {");
        in();
        for (auto param : *o->params) {
          auto name = dynamic_cast<const Parameter*>(param)->name;
          std::string flag = sanitize(name->str()) + "FLAG_";
          std::string option = std::regex_replace(name->str(), std::regex("_"), "-");

          genSourceLine(o->loc);
          start("{\"");
          middle(option << "\", required_argument, 0, " << flag);
          finish(" },");
        }
        genSourceLine(o->loc);
        line("{0, 0, 0, 0}");
        out();
        genSourceLine(o->loc);
        line("};");

        /* short options */
        genSourceLine(o->loc);
        line("const char* short_options_ = \":\";");

        /* handle error reporting ourselves (worth commenting this out if
        * debugging issues with command-line parsing) */
        genSourceLine(o->loc);
        line("::opterr = 0;");

        /* read in options with getopt_long */
        genSourceLine(o->loc);
        start("c_ = ::getopt_long_only(argc_, argv_, short_options_, ");
        finish("long_options_, &option_index_);");
        genSourceLine(o->loc);
        line("while (c_ != -1) {");
        in();
        genSourceLine(o->loc);
        line("switch (c_) {");
        in();

        for (auto param : *o->params) {
          auto p = dynamic_cast<const Parameter*>(param);
          auto name = p->name;
          auto flag = sanitize(name->str()) + "FLAG_";

          genSourceLine(p->loc);
          line("case " << flag << ':');
          in();
          genSourceLine(p->loc);
          line("if (!::optarg) {");
          in();
          genSourceLine(p->loc);
          line("error(String(\"value required for option --\") + String(long_options_[::optopt].name));");
          out();
          genSourceLine(p->loc);
          line("}");
          genSourceLine(p->loc);
          auto type = dynamic_cast<Named*>(p->type->unwrap());
          if (type && type->name->str() != "String") {
            start(name << " = from_string<" << type->name << '>');
            finish("(String(::optarg));");
          } else {
            line(name << " = String(::optarg);");
          }
          genSourceLine(p->loc);
          line("break;");
          out();
        }

        genSourceLine(o->loc);
        line("case '?':");
        in();
        genSourceLine(o->loc);
        line("error(String(\"unrecognized option \") + String(argv_[::optind - 1]));");
        out();

        genSourceLine(o->loc);
        line("case ':':");
        in();
        genSourceLine(o->loc);
        line("error(String(\"value required for option --\") + String(long_options_[::optopt].name));");
        out();

        genSourceLine(o->loc);
        line("default:");
        in();
        genSourceLine(o->loc);
        line("error(String(\"unknown error parsing command-line options.\"));");
        out();

        out();
        line('}');
        genSourceLine(o->loc);
        start("c_ = ::getopt_long_only(argc_, argv_, short_options_, ");
        finish("long_options_, &option_index_);");
        out();
        line("}\n");
      }

      /* initialization */
      genSourceLine(o->loc);
      line("numbirch::init();\n");
      genSourceLine(o->loc);
      line("birch::init();\n");

      /* body of program */
      *this << o->braces;

      /* termination */
      genSourceLine(o->loc);
      line("birch::term();\n");
      genSourceLine(o->loc);
      line("membirch::collect();");
      genSourceLine(o->loc);
      line("numbirch::term();\n");
      genSourceLine(o->loc);
      line("return 0;");
      out();
      line("}\n");

      /* register program */
      if (!o->braces->isEmpty()) {
        start("static int register_program_" << o->name);
        middle("_ = ::register_program(");
        finish("\"" << o->name << "\", " << o->name << ");");
      }
    }
  }
}

void birch::CppGenerator::visit(const BinaryOperator* o) {
  if ((includeInline || !o->isGeneric()) && !o->braces->isEmpty()) {
    genDoc(o->loc);
    genTemplateParams(o);
    genSourceLine(o->loc);
    start("");
    if (!header && o->isGeneric()) {
      middle("inline ");
    }
    middle(o->returnType << ' ');
    if (isTranslatable(o->name->str())) {
      middle("operator" << o->name->str());
    } else {
      middle(o->name);
    }
    middle('(' << o->left << ", " << o->right << ')');
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      ++inOperator;
      *this << o->braces->strip();
      --inOperator;
      out();
      finish("}\n");
    }
  }
}

void birch::CppGenerator::visit(const UnaryOperator* o) {
  if ((includeInline || !o->isGeneric()) && !o->braces->isEmpty()) {
    genDoc(o->loc);
    genTemplateParams(o);
    genSourceLine(o->loc);
    start(o->returnType << ' ');
    if (isTranslatable(o->name->str())) {
      middle("operator" << o->name->str());
    } else {
      middle(o->name);
    }
    middle('(' << o->single << ')');
    if (header) {
      finish(';');
    } else {
      finish(" {");
      in();
      ++inOperator;
      *this << o->braces->strip();
      --inOperator;
      out();
      finish("}\n");
    }
  }
}

void birch::CppGenerator::visit(const Basic* o) {
  //
}

void birch::CppGenerator::visit(const Class* o) {
  if (includeInline || !o->isGeneric()) {
    CppClassGenerator auxClass(base, level, header, includeInline,
        includeLines, o);
    auxClass << o;
  }
}

void birch::CppGenerator::visit(const Struct* o) {
  if (includeInline || !o->isGeneric()) {
    CppStructGenerator auxStruct(base, level, header, includeInline,
        includeLines, o);
    auxStruct << o;
  }
}

void birch::CppGenerator::visit(const Generic* o) {
  middle(o->name);
}

void birch::CppGenerator::visit(const Braces* o) {
  line('{');
  in();
  *this << o->single;
  out();
  line('}');
}

void birch::CppGenerator::visit(const Factor* o) {
  genSourceLine(o->loc);
  line("handle_factor(" << o->single << ");");
}

void birch::CppGenerator::visit(const ExpressionStatement* o) {
  genSourceLine(o->loc);
  line(o->single << ';');
}

void birch::CppGenerator::visit(const If* o) {
  genSourceLine(o->loc);
  line("if (" << o->cond->strip() << ") {");
  in();
  *this << o->braces->strip();
  out();
  if (!o->falseBraces->isEmpty()) {
    line("} else {");
    in();
    *this << o->falseBraces->strip();
    out();
  }
  line("}");
}

void birch::CppGenerator::visit(const For* o) {
  auto index = genIndex(o->index);
  genSourceLine(o->loc);
  start("for (int " << index << " = int(" << o->from << "); ");
  finish(index << " <= int(" << o->to << "); ++" << index << ") {");
  in();
  *this << o->braces->strip();
  out();
  line("}");
}

void birch::CppGenerator::visit(const Parallel* o) {
  auto index = genIndex(o->index);
  genSourceLine(o->loc);
  line("#if HAVE_NUMBIRCH_HPP");
  line("numbirch::wait();");
  line("#endif");
  line("#pragma omp parallel");
  line("{");
  in();
  start("#pragma omp for schedule(");
  if (o->has(DYNAMIC)) {
    middle("guided");
  } else {
    middle("static");
  }
  finish(')');
  start("for (int " << index << " = int(" << o->from << "); ");
  finish(index << " <= int(" << o->to << "); ++" << index << ") {");
  in();
  *this << o->braces->strip();
  out();
  line("}");
  line("#if HAVE_NUMBIRCH_HPP");
  line("numbirch::wait();");
  line("#endif");
  out();
  line("}");
}

void birch::CppGenerator::visit(const While* o) {
  genSourceLine(o->loc);
  line("while (" << o->cond->strip() << ") {");
  in();
  *this << o->braces->strip();
  out();
  line("}");
}

void birch::CppGenerator::visit(const DoWhile* o) {
  genSourceLine(o->loc);
  line("do {");
  in();
  *this << o->braces->strip();
  out();
  line("} while (" << o->cond->strip() << ");");
}

void birch::CppGenerator::visit(const With* o) {
  genSourceLine(o->loc);
  line("{");
  in();
  line("auto handler_ = swap_handler(" << o->single << ");");
  *this << o->braces->strip();
  line("set_handler(handler_);");
  out();
  line("}");
}

void birch::CppGenerator::visit(const Assert* o) {
  genSourceLine(o->loc);
  line("assert(" << o->cond->strip() << ");");
}

void birch::CppGenerator::visit(const Return* o) {
  genSourceLine(o->loc);
  ++inReturn;
  line("return " << o->single << ';');
  --inReturn;
}

void birch::CppGenerator::visit(const Raw* o) {
  if ((header && *o->name == "hpp") || (!header && *o->name == "cpp")) {
    genSourceLine(o->loc);
    *this << escape_unicode(o->raw);
    if (!std::isspace(o->raw.back())) {
      *this << ' ';
    }
  }
}

void birch::CppGenerator::visit(const StatementList* o) {
  middle(o->head << o->tail);
}

void birch::CppGenerator::visit(const EmptyType* o) {
  middle("void");
}

void birch::CppGenerator::visit(const ArrayType* o) {
  middle("numbirch::Array<" << o->single << ',' << o->depth() << '>');
}

void birch::CppGenerator::visit(const TupleType* o) {
  middle("std::tuple<" << o->single << '>');
}

void birch::CppGenerator::visit(const OptionalType* o) {
  middle("std::optional<" << o->single << '>');
}

void birch::CppGenerator::visit(const FutureType* o) {
  middle("numbirch::Future<" << o->single << '>');
}

void birch::CppGenerator::visit(const MemberType* o) {
  middle("typename " << o->left << "::value_type::" << o->right << "_");
}

void birch::CppGenerator::visit(const NamedType* o) {
  middle(o->name);
  if (!o->typeArgs->isEmpty()) {
    middle('<' << o->typeArgs << '>');
  }
}

void birch::CppGenerator::visit(const TypeList* o) {
  middle(o->head << ", " << o->tail);
}

void birch::CppGenerator::visit(const DeducedType* o) {
  middle("auto");
}

void birch::CppGenerator::genDoc(const Location* loc) {
  if (!loc->doc.empty()) {
    line("");
    line("/**" << loc->doc << "*/");
  }
}

void birch::CppGenerator::genSourceLine(const Location* loc) {
  if (includeLines) {
    auto line = loc->firstLine;
    auto file = loc->file->path;
    line("#line " << line << " \"" << file << "\"");
  }
}

std::string birch::CppGenerator::genIndex(const Statement* o) {
  auto index = dynamic_cast<const LocalVariable*>(o);
  assert(index);
  return sanitize(index->name->str());
}
