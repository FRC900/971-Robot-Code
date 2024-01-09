#ifndef AOS_FLATBUFFER_UTILS_
#define AOS_FLATBUFFER_UTILS_

#include <optional>
#include <string_view>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/reflection_generated.h"
#include "glog/logging.h"

namespace aos {

// Returns a human readable description of the type.
inline const char *ElementaryTypeName(
    const flatbuffers::ElementaryType elementary_type) {
  return flatbuffers::ElementaryTypeNames()[elementary_type] + 3;
}

// A least-common-denominator API for a TypeTable or a Schema.
//
// An instance may represent an enum or a sequence (table, struct, or union).
// Schemas have objects for the individual fields, but these are not exposed as
// FlatbufferType instances.
class FlatbufferType final {
 public:
  // Implicit on purpose, to allow freely creating a FlatbufferType.
  FlatbufferType(const flatbuffers::TypeTable *type_table)
      : type_table_(CHECK_NOTNULL(type_table)) {}
  FlatbufferType(const reflection::Schema *schema)
      : schema_(CHECK_NOTNULL(schema)),
        object_(DCHECK_NOTNULL(schema->root_table())) {}

  // This is deliberately copyable, for ease of memory management. It is cheap
  // to pass by value.
  FlatbufferType(const FlatbufferType &) = default;
  FlatbufferType(FlatbufferType &&) = default;
  FlatbufferType &operator=(const FlatbufferType &) = default;
  FlatbufferType &operator=(FlatbufferType &&) = default;

  // Returns whether this type is a sequence (table, struct, or union).
  bool IsSequence() const;

  // Returns whether this type is an enum.
  bool IsEnum() const;

  // Returns whether this type is a struct.
  bool IsStruct() const;

  // Returns whether this type is a table.
  bool IsTable() const;

  // Returns whether the given field is a sequence (table, struct, or union).
  //
  // Only valid for sequences (tables, structs, or unions).
  bool FieldIsSequence(int index) const;

  // Returns whether the given field is an enum.
  //
  // Only valid for sequences (tables, structs, or unions).
  bool FieldIsEnum(int index) const;

  // Returns the value for a given enumerator.
  //
  // Only valid for enums.
  std::optional<int64_t> EnumValue(std::string_view name) const;

  // Returns whether the given field is either a vector (in a table) or an array
  // (in a struct).
  //
  // Only valid for sequences (tables, structs, or unions).
  bool FieldIsRepeating(int index) const;

  // Returns the field index in a table given the name, or -1 if the name is not
  // found.
  //
  // Only valid for sequences (tables, structs, or unions).
  int FieldIndex(std::string_view field_name) const;

  // Returns the name for a field.
  //
  // Only valid for sequences (tables, structs, or unions).
  std::string_view FieldName(int index) const;

  // Returns the type of a field.
  //
  // Only valid for sequences (tables, structs, or unions).
  flatbuffers::ElementaryType FieldElementaryType(int index) const;

  // See flatbuffers::InlineSize for details.
  //
  // Only valid for sequences (tables, structs, or unions).
  size_t FieldInlineSize(int index) const;
  size_t InlineSize() const;

  // Returns the total number of fields.
  //
  // Only valid for sequences (tables, structs, or unions).
  int NumberFields() const;

  // Returns the type for a field.
  //
  // Only valid for sequences (tables, structs, or unions).
  FlatbufferType FieldType(int index) const;

  // Returns the offset of the specified field within the struct.
  //
  // Only valid for structs.
  size_t StructFieldOffset(int index) const;

  // Returns the required alignment for this type.
  size_t Alignment() const;
  // The required alignment of the inline data for the specified field.
  size_t FieldInlineAlignment(size_t field_index) const;

 private:
  explicit FlatbufferType(const reflection::Schema *schema,
                          const reflection::Object *object)
      : schema_(DCHECK_NOTNULL(schema)), object_(DCHECK_NOTNULL(object)) {}
  explicit FlatbufferType(const reflection::Schema *schema,
                          const reflection::Enum *fb_enum)
      : schema_(DCHECK_NOTNULL(schema)), enum_(DCHECK_NOTNULL(fb_enum)) {}

  const reflection::Type *ReflectionType(int index) const;
  const reflection::Field *ReflectionObjectField(int index) const;
  const reflection::EnumVal *ReflectionEnumValue(int index) const;
  reflection::BaseType ReflectionElementBaseType(int index) const;

  const flatbuffers::TypeTable *type_table_ = nullptr;
  const reflection::Schema *schema_ = nullptr;
  const reflection::Object *object_ = nullptr;
  const reflection::Enum *enum_ = nullptr;
};

}  // namespace aos

#endif  // AOS_FLATBUFFER_UTILS_
