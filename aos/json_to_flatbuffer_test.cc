#include "aos/json_to_flatbuffer.h"

#include "aos/json_to_flatbuffer_generated.h"
#include "aos/testing/path.h"
#include "flatbuffers/minireflect.h"
#include "gtest/gtest.h"

namespace aos {
namespace testing {

class JsonToFlatbufferTest : public ::testing::Test {
 public:
  JsonToFlatbufferTest() {}

  FlatbufferVector<reflection::Schema> Schema() {
    return FileToFlatbuffer<reflection::Schema>(
        ArtifactPath("aos/json_to_flatbuffer.bfbs"));
  }

  bool JsonAndBack(const ::std::string str) { return JsonAndBack(str, str); }

  bool JsonAndBack(const ::std::string in, const ::std::string out) {
    printf("Testing: %s\n", in.c_str());
    FlatbufferDetachedBuffer<Configuration> fb_typetable =
        JsonToFlatbuffer<Configuration>(in.data());
    FlatbufferDetachedBuffer<Configuration> fb_reflection =
        JsonToFlatbuffer(in.data(), FlatbufferType(&Schema().message()));

    if (fb_typetable.span().size() == 0) {
      return false;
    }
    if (fb_reflection.span().size() == 0) {
      return false;
    }

    const ::std::string back_typetable = FlatbufferToJson(fb_typetable);
    const ::std::string back_reflection = FlatbufferToJson(fb_reflection);

    printf("Back to string via TypeTable: %s\n", back_typetable.c_str());
    printf("Back to string via reflection: %s\n", back_reflection.c_str());

    return back_typetable == out && back_reflection == out;
  }
};

// Tests that the various escapes work as expected.
TEST_F(JsonToFlatbufferTest, ValidEscapes) {
  EXPECT_TRUE(
      JsonAndBack("{ \"foo_string\": \"a\\\"b\\/c\\bd\\fc\\nd\\re\\tf\" }",
                  "{ \"foo_string\": \"a\\\"b/c\\bd\\fc\\nd\\re\\tf\" }"));
}

// Test the easy ones.  Test every type, single, no nesting.
TEST_F(JsonToFlatbufferTest, Basic) {
  EXPECT_TRUE(JsonAndBack("{ \"foo_bool\": true }"));

  EXPECT_TRUE(JsonAndBack("{ \"foo_byte\": 5 }"));
  EXPECT_TRUE(JsonAndBack("{ \"foo_ubyte\": 5 }"));

  EXPECT_TRUE(JsonAndBack("{ \"foo_short\": 5 }"));
  EXPECT_TRUE(JsonAndBack("{ \"foo_ushort\": 5 }"));

  EXPECT_TRUE(JsonAndBack("{ \"foo_int\": 5 }"));
  EXPECT_TRUE(JsonAndBack("{ \"foo_uint\": 5 }"));

  EXPECT_TRUE(JsonAndBack("{ \"foo_long\": 5 }"));
  EXPECT_TRUE(JsonAndBack("{ \"foo_ulong\": 5 }"));

  EXPECT_TRUE(JsonAndBack("{ \"foo_float\": 5.0 }"));
  EXPECT_TRUE(JsonAndBack("{ \"foo_double\": 5.0 }"));

  EXPECT_TRUE(JsonAndBack("{ \"foo_enum\": \"None\" }"));
  EXPECT_TRUE(JsonAndBack("{ \"foo_enum\": \"UType\" }"));

  EXPECT_TRUE(JsonAndBack("{ \"foo_enum_default\": \"None\" }"));
  EXPECT_TRUE(JsonAndBack("{ \"foo_enum_default\": \"UType\" }"));

  EXPECT_TRUE(JsonAndBack("{ \"foo_string\": \"baz\" }"));

  EXPECT_TRUE(JsonAndBack("{ \"foo_enum_nonconsecutive\": \"Zero\" }"));
  EXPECT_TRUE(JsonAndBack("{ \"foo_enum_nonconsecutive\": \"Big\" }"));
}

// Tests that Inf is handled correctly
TEST_F(JsonToFlatbufferTest, Inf) {
  EXPECT_TRUE(JsonAndBack("{ \"foo_float\": inf }"));
  EXPECT_TRUE(JsonAndBack("{ \"foo_float\": -inf }"));
  EXPECT_TRUE(JsonAndBack("{ \"foo_double\": inf }"));
  EXPECT_TRUE(JsonAndBack("{ \"foo_double\": -inf }"));
}

// Tests that NaN is handled correctly
TEST_F(JsonToFlatbufferTest, Nan) {
  EXPECT_TRUE(JsonAndBack("{ \"foo_float\": nan }"));
  EXPECT_TRUE(JsonAndBack("{ \"foo_float\": -nan }"));
  EXPECT_TRUE(JsonAndBack("{ \"foo_double\": nan }"));
  EXPECT_TRUE(JsonAndBack("{ \"foo_double\": -nan }"));
}

// Tests that unicode is handled correctly
TEST_F(JsonToFlatbufferTest, Unicode) {
  EXPECT_TRUE(JsonAndBack("{ \"foo_string\": \"\\uF672\" }"));
  EXPECT_TRUE(JsonAndBack("{ \"foo_string\": \"\\uEFEF\" }"));
  EXPECT_TRUE(JsonAndBack("{ \"foo_string\": \"helloworld\\uD83E\\uDE94\" }"));
  EXPECT_TRUE(JsonAndBack("{ \"foo_string\": \"\\uD83C\\uDF32\" }"));
  EXPECT_FALSE(JsonAndBack("{ \"foo_string\": \"\\uP890\" }"));
  EXPECT_FALSE(JsonAndBack("{ \"foo_string\": \"\\u!FA8\" }"));
  EXPECT_FALSE(JsonAndBack("{ \"foo_string\": \"\\uF89\" }"));
  EXPECT_FALSE(JsonAndBack("{ \"foo_string\": \"\\uD83C\" }"));
}

// Tests that we can handle decimal points.
TEST_F(JsonToFlatbufferTest, DecimalPoint) {
  EXPECT_TRUE(JsonAndBack("{ \"foo_float\": 5.1 }"));
  EXPECT_TRUE(JsonAndBack("{ \"foo_double\": 5.1 }"));
}

// Test what happens if you pass a field name that we don't know.
TEST_F(JsonToFlatbufferTest, InvalidFieldName) {
  EXPECT_FALSE(JsonAndBack("{ \"foo\": 5 }"));
}

// Tests that an invalid enum type is handled correctly.
TEST_F(JsonToFlatbufferTest, InvalidEnumName) {
  EXPECT_FALSE(JsonAndBack("{ \"foo_enum\": \"5ype\" }"));

  EXPECT_FALSE(JsonAndBack("{ \"foo_enum_default\": \"7ype\" }"));

  EXPECT_FALSE(JsonAndBack("{ \"foo_enum_nonconsecutive\": \"Nope\" }"));

  EXPECT_FALSE(
      JsonAndBack("{ \"foo_enum_nonconsecutive_default\": \"Nope\" }"));
}

// Test that adding a duplicate field results in an error.
TEST_F(JsonToFlatbufferTest, DuplicateField) {
  EXPECT_FALSE(
      JsonAndBack("{ \"foo_int\": 5, \"foo_int\": 7 }", "{ \"foo_int\": 7 }"));
}

// Test that various syntax errors are caught correctly
TEST_F(JsonToFlatbufferTest, InvalidSyntax) {
  EXPECT_FALSE(JsonAndBack("{ \"foo_int\": 5"));
  EXPECT_FALSE(JsonAndBack("{ \"foo_int\": 5 "));
  EXPECT_FALSE(JsonAndBack("{ \"foo_string\": \""));
  EXPECT_FALSE(JsonAndBack("{ \"foo_int\": 5 } }"));

  EXPECT_FALSE(JsonAndBack("{ foo_int: 5 }"));

  EXPECT_FALSE(JsonAndBack("{ \"foo_int\": 5, }", "{ \"foo_int\": 5 }"));

  EXPECT_FALSE(
      JsonAndBack("{ \"apps\":\n[\n{\n\"name\": \"woot\"\n},\n{\n\"name\": "
                  "\"wow\"\n} ,\n]\n}"));

  EXPECT_FALSE(JsonAndBack(
      "{ \"apps\": [ { \"name\": \"woot\" }, { \"name\": \"wow\" } ] , }"));

  EXPECT_FALSE(
      JsonAndBack("{ \"vector_foo_string\": [ \"bar\", \"baz\" ] , }"));

  EXPECT_FALSE(
      JsonAndBack("{ \"single_application\": { \"name\": \"woot\" } , }"));
}

// Test arrays of simple types.
TEST_F(JsonToFlatbufferTest, Array) {
  EXPECT_TRUE(JsonAndBack("{ \"vector_foo_byte\": [ 9, 7, 1 ] }"));
  EXPECT_TRUE(JsonAndBack("{ \"vector_foo_byte\": [  ] }"));
  EXPECT_TRUE(JsonAndBack("{ \"vector_foo_ubyte\": [ 9, 7, 1 ] }"));
  EXPECT_TRUE(JsonAndBack("{ \"vector_foo_ubyte\": [  ] }"));

  EXPECT_TRUE(JsonAndBack("{ \"vector_foo_short\": [ 9, 7, 1 ] }"));
  EXPECT_TRUE(JsonAndBack("{ \"vector_foo_short\": [  ] }"));
  EXPECT_TRUE(JsonAndBack("{ \"vector_foo_ushort\": [ 9, 7, 1 ] }"));
  EXPECT_TRUE(JsonAndBack("{ \"vector_foo_ushort\": [  ] }"));

  EXPECT_TRUE(JsonAndBack("{ \"vector_foo_int\": [ 9, 7, 1 ] }"));
  EXPECT_TRUE(JsonAndBack("{ \"vector_foo_int\": [  ] }"));
  EXPECT_TRUE(JsonAndBack("{ \"vector_foo_uint\": [ 9, 7, 1 ] }"));
  EXPECT_TRUE(JsonAndBack("{ \"vector_foo_uint\": [  ] }"));

  EXPECT_TRUE(JsonAndBack("{ \"vector_foo_long\": [ 9, 7, 1 ] }"));
  EXPECT_TRUE(JsonAndBack("{ \"vector_foo_long\": [  ] }"));
  EXPECT_TRUE(JsonAndBack("{ \"vector_foo_ulong\": [ 9, 7, 1 ] }"));
  EXPECT_TRUE(JsonAndBack("{ \"vector_foo_ulong\": [  ] }"));

  EXPECT_TRUE(JsonAndBack("{ \"vector_foo_float\": [ 9.0, 7.0, 1.0 ] }"));
  EXPECT_TRUE(JsonAndBack("{ \"vector_foo_float\": [  ] }"));
  EXPECT_TRUE(JsonAndBack("{ \"vector_foo_double\": [ 9.0, 7.0, 1.0 ] }"));
  EXPECT_TRUE(JsonAndBack("{ \"vector_foo_double\": [  ] }"));

  EXPECT_TRUE(JsonAndBack("{ \"vector_foo_float\": [ 9, 7, 1 ] }",
                          "{ \"vector_foo_float\": [ 9.0, 7.0, 1.0 ] }"));
  EXPECT_TRUE(JsonAndBack("{ \"vector_foo_double\": [ 9, 7, 1 ] }",
                          "{ \"vector_foo_double\": [ 9.0, 7.0, 1.0 ] }"));

  EXPECT_TRUE(JsonAndBack("{ \"vector_foo_string\": [ \"bar\", \"baz\" ] }"));
  EXPECT_TRUE(JsonAndBack("{ \"vector_foo_string\": [  ] }"));
  EXPECT_TRUE(JsonAndBack(
      "{ \"vector_foo_enum\": [ \"None\", \"UType\", \"Bool\" ] }"));
  EXPECT_TRUE(JsonAndBack("{ \"vector_foo_enum\": [  ] }"));
}

// Test nested messages, and arrays of nested messages.
TEST_F(JsonToFlatbufferTest, NestedStruct) {
  EXPECT_TRUE(
      JsonAndBack("{ \"single_application\": { \"name\": \"woot\" } }"));

  EXPECT_TRUE(JsonAndBack("{ \"single_application\": {  } }"));

  EXPECT_TRUE(JsonAndBack(
      "{ \"apps\": [ { \"name\": \"woot\" }, { \"name\": \"wow\" } ] }"));

  EXPECT_TRUE(JsonAndBack("{ \"apps\": [ {  }, {  } ] }"));
}

// Test mixing up whether a field is an object or a vector.
TEST_F(JsonToFlatbufferTest, IncorrectVectorOfTables) {
  EXPECT_FALSE(
      JsonAndBack("{ \"single_application\": [ {\"name\": \"woot\"} ] }"));
  EXPECT_FALSE(JsonAndBack("{ \"apps\": { \"name\": \"woot\" } }"));
}

// Test that we can parse an empty message.
TEST_F(JsonToFlatbufferTest, EmptyMessage) {
  // Empty message works.
  EXPECT_TRUE(JsonAndBack("{  }"));
}

// Tests that comments get stripped.
TEST_F(JsonToFlatbufferTest, Comments) {
  EXPECT_TRUE(JsonAndBack("{ /* foo */ \"vector_foo_double\": [ 9, 7, 1 ] }",
                          "{ \"vector_foo_double\": [ 9.0, 7.0, 1.0 ] }"));
}

// Tests that multiple arrays get properly handled.
TEST_F(JsonToFlatbufferTest, MultipleArrays) {
  EXPECT_TRUE(
      JsonAndBack("{ \"vector_foo_float\": [ 9, 7, 1 ], \"vector_foo_double\": "
                  "[ 9, 7, 1 ] }",
                  "{ \"vector_foo_float\": [ 9.0, 7.0, 1.0 ], "
                  "\"vector_foo_double\": [ 9.0, 7.0, 1.0 ] }"));
}

// Tests that multiple arrays get properly handled.
TEST_F(JsonToFlatbufferTest, NestedArrays) {
  EXPECT_TRUE(
      JsonAndBack("{ \"vov\": { \"v\": [ { \"str\": [ \"a\", \"b\" ] }, { "
                  "\"str\": [ \"c\", \"d\" ] } ] } }"));
}

// TODO(austin): Missmatched values.
//
// TODO(austin): unions?

TEST_F(JsonToFlatbufferTest, TrimmedVector) {
  std::string json_short = "{ \"vector_foo_int\": [ 0";
  for (int i = 1; i < 100; ++i) {
    json_short += ", ";
    json_short += std::to_string(i);
  }
  std::string json_long = json_short;
  json_short += " ] }";
  json_long += ", 101 ] }";

  const FlatbufferDetachedBuffer<Configuration> fb_short_typetable(
      JsonToFlatbuffer<Configuration>(json_short));
  ASSERT_GT(fb_short_typetable.span().size(), 0);
  const FlatbufferDetachedBuffer<Configuration> fb_long_typetable(
      JsonToFlatbuffer<Configuration>(json_long));
  ASSERT_GT(fb_long_typetable.span().size(), 0);
  const FlatbufferDetachedBuffer<Configuration> fb_short_reflection(
      JsonToFlatbuffer(json_short, FlatbufferType(&Schema().message())));
  ASSERT_GT(fb_short_reflection.span().size(), 0);
  const FlatbufferDetachedBuffer<Configuration> fb_long_reflection(
      JsonToFlatbuffer(json_long, FlatbufferType(&Schema().message())));
  ASSERT_GT(fb_long_reflection.span().size(), 0);

  const std::string back_json_short_typetable = FlatbufferToJson<Configuration>(
      fb_short_typetable, {.multi_line = false, .max_vector_size = 100});
  const std::string back_json_long_typetable = FlatbufferToJson<Configuration>(
      fb_long_typetable, {.multi_line = false, .max_vector_size = 100});
  const std::string back_json_short_reflection =
      FlatbufferToJson<Configuration>(
          fb_short_reflection, {.multi_line = false, .max_vector_size = 100});
  const std::string back_json_long_reflection = FlatbufferToJson<Configuration>(
      fb_long_reflection, {.multi_line = false, .max_vector_size = 100});

  EXPECT_EQ(json_short, back_json_short_typetable);
  EXPECT_EQ(json_short, back_json_short_reflection);
  EXPECT_EQ("{ \"vector_foo_int\": [ \"... 101 elements ...\" ] }",
            back_json_long_typetable);
  EXPECT_EQ("{ \"vector_foo_int\": [ \"... 101 elements ...\" ] }",
            back_json_long_reflection);
}

// Tests that a nullptr buffer prints nullptr.
TEST_F(JsonToFlatbufferTest, NullptrData) {
  EXPECT_EQ("null", TableFlatbufferToJson((const flatbuffers::Table *)(nullptr),
                                          ConfigurationTypeTable()));
}

}  // namespace testing
}  // namespace aos
