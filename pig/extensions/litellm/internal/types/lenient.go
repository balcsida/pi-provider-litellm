package types

import (
	"encoding/json"
	"reflect"
	"strings"
)

// The TypeScript reads /model/info fields as unknown and validates each one (wireString,
// wireBoolean, Array.isArray), so one mistyped field is ignored instead of failing the whole
// response. The UnmarshalJSON methods below do the same: a field whose JSON type does not match
// stays at its zero value, and a non-object value decodes to the zero struct.

// malformedListIsEmpty names the list fields where a present but non-list value is not an
// absence: the TypeScript treats it as a list that names nothing (no carrier evidence, no
// Messages endpoint), while null and a missing key stay nil (unknown).
var malformedListIsEmpty = map[string]bool{
	"supported_endpoints":     true,
	"supported_openai_params": true,
	"allowed_openai_params":   true,
}

func decodeLenient(data []byte, dst any) error {
	var fields map[string]json.RawMessage
	if json.Unmarshal(data, &fields) != nil || fields == nil {
		return nil
	}
	v := reflect.ValueOf(dst).Elem()
	t := v.Type()
	for i := range t.NumField() {
		name, _, _ := strings.Cut(t.Field(i).Tag.Get("json"), ",")
		raw, ok := fields[name]
		if !ok || string(raw) == "null" {
			continue
		}
		field := v.Field(i)
		if field.Kind() == reflect.Slice {
			var elements []json.RawMessage
			if json.Unmarshal(raw, &elements) != nil {
				if malformedListIsEmpty[name] {
					field.Set(reflect.ValueOf([]string{}))
				}
				continue
			}
			// Non-string elements are dropped, as the TypeScript filters them with wireString.
			list := make([]string, 0, len(elements))
			for _, element := range elements {
				var s string
				if json.Unmarshal(element, &s) == nil && string(element) != "null" {
					list = append(list, s)
				}
			}
			field.Set(reflect.ValueOf(list))
			continue
		}
		decoded := reflect.New(field.Type())
		if json.Unmarshal(raw, decoded.Interface()) == nil {
			field.Set(decoded.Elem())
		}
	}
	return nil
}

func (p *ModelInfoParams) UnmarshalJSON(data []byte) error {
	type plain ModelInfoParams
	*p = ModelInfoParams{}
	return decodeLenient(data, (*plain)(p))
}

func (d *ModelInfoDetails) UnmarshalJSON(data []byte) error {
	type plain ModelInfoDetails
	*d = ModelInfoDetails{}
	return decodeLenient(data, (*plain)(d))
}

func (e *ModelInfoEntry) UnmarshalJSON(data []byte) error {
	type plain ModelInfoEntry
	*e = ModelInfoEntry{}
	return decodeLenient(data, (*plain)(e))
}
