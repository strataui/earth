//! Minimal deterministic JSON reader and writer.
//!
//! The golden descriptor is exchanged with external reference viewports, so it
//! has to be real JSON. The workspace policy forbids heavyweight crates, and a
//! descriptor only needs objects, arrays, strings, numbers, booleans, and null;
//! that is the entire surface implemented here.
//!
//! Objects keep insertion order so serializing the same descriptor twice
//! produces byte-identical output.

use std::fmt::Write;

/// One JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn object(entries: Vec<(&str, Json)>) -> Self {
        Self::Object(
            entries
                .into_iter()
                .map(|(key, value)| (key.to_string(), value))
                .collect(),
        )
    }

    pub fn number(value: impl Into<f64>) -> Self {
        Self::Number(value.into())
    }

    pub fn array(values: Vec<Json>) -> Self {
        Self::Array(values)
    }

    pub fn numbers(values: &[f64]) -> Self {
        Self::Array(values.iter().copied().map(Self::Number).collect())
    }

    pub fn string(value: impl Into<String>) -> Self {
        Self::String(value.into())
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Self::Object(entries) => entries
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Self::Array(values) => Some(values),
            _ => None,
        }
    }

    /// Parses a JSON document.
    pub fn parse(text: &str) -> Result<Self, String> {
        let bytes = text.as_bytes();
        let mut cursor = 0usize;
        let value = parse_value(bytes, &mut cursor)?;
        skip_whitespace(bytes, &mut cursor);
        if cursor != bytes.len() {
            return Err(format!("trailing data at byte {cursor}"));
        }
        Ok(value)
    }

    /// Serializes to compact JSON.
    pub fn to_json_string(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }
}

impl std::fmt::Display for Json {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out = String::new();
        self.write(&mut out);
        write!(f, "{out}")
    }
}

impl Json {

    fn write(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
            Self::Number(value) => write_number(*value, out),
            Self::String(value) => write_string(value, out),
            Self::Array(values) => {
                out.push('[');
                for (index, value) in values.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    value.write(out);
                }
                out.push(']');
            }
            Self::Object(entries) => {
                out.push('{');
                for (index, (key, value)) in entries.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    write_string(key, out);
                    out.push(':');
                    value.write(out);
                }
                out.push('}');
            }
        }
    }
}

fn write_number(value: f64, out: &mut String) {
    if !value.is_finite() {
        out.push_str("null");
        return;
    }
    if value == 0.0 {
        out.push('0');
        return;
    }
    // Rust's Display already emits the shortest round-tripping form and omits
    // a trailing `.0` for integral values.
    let _ = write!(out, "{value}");
}

fn write_string(value: &str, out: &mut String) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if (ch as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", ch as u32);
            }
            ch => out.push(ch),
        }
    }
    out.push('"');
}

fn skip_whitespace(bytes: &[u8], cursor: &mut usize) {
    while *cursor < bytes.len() && matches!(bytes[*cursor], b' ' | b'\t' | b'\n' | b'\r') {
        *cursor += 1;
    }
}

fn expect(bytes: &[u8], cursor: &mut usize, expected: u8) -> Result<(), String> {
    if bytes.get(*cursor) == Some(&expected) {
        *cursor += 1;
        Ok(())
    } else {
        Err(format!(
            "expected '{}' at byte {}",
            expected as char, *cursor
        ))
    }
}

fn parse_value(bytes: &[u8], cursor: &mut usize) -> Result<Json, String> {
    skip_whitespace(bytes, cursor);
    match bytes.get(*cursor) {
        Some(b'{') => parse_object(bytes, cursor),
        Some(b'[') => parse_array(bytes, cursor),
        Some(b'"') => parse_string(bytes, cursor).map(Json::String),
        Some(b't') => parse_literal(bytes, cursor, b"true", Json::Bool(true)),
        Some(b'f') => parse_literal(bytes, cursor, b"false", Json::Bool(false)),
        Some(b'n') => parse_literal(bytes, cursor, b"null", Json::Null),
        Some(_) => parse_number(bytes, cursor),
        None => Err("unexpected end of input".to_string()),
    }
}

fn parse_literal(
    bytes: &[u8],
    cursor: &mut usize,
    literal: &[u8],
    value: Json,
) -> Result<Json, String> {
    if bytes[*cursor..].starts_with(literal) {
        *cursor += literal.len();
        Ok(value)
    } else {
        Err(format!("invalid literal at byte {cursor}"))
    }
}

fn parse_object(bytes: &[u8], cursor: &mut usize) -> Result<Json, String> {
    expect(bytes, cursor, b'{')?;
    let mut entries = Vec::new();
    skip_whitespace(bytes, cursor);
    if bytes.get(*cursor) == Some(&b'}') {
        *cursor += 1;
        return Ok(Json::Object(entries));
    }
    loop {
        skip_whitespace(bytes, cursor);
        let key = parse_string(bytes, cursor)?;
        skip_whitespace(bytes, cursor);
        expect(bytes, cursor, b':')?;
        let value = parse_value(bytes, cursor)?;
        entries.push((key, value));
        skip_whitespace(bytes, cursor);
        match bytes.get(*cursor) {
            Some(b',') => {
                *cursor += 1;
            }
            Some(b'}') => {
                *cursor += 1;
                return Ok(Json::Object(entries));
            }
            _ => return Err(format!("expected ',' or '}}' at byte {cursor}")),
        }
    }
}

fn parse_array(bytes: &[u8], cursor: &mut usize) -> Result<Json, String> {
    expect(bytes, cursor, b'[')?;
    let mut values = Vec::new();
    skip_whitespace(bytes, cursor);
    if bytes.get(*cursor) == Some(&b']') {
        *cursor += 1;
        return Ok(Json::Array(values));
    }
    loop {
        values.push(parse_value(bytes, cursor)?);
        skip_whitespace(bytes, cursor);
        match bytes.get(*cursor) {
            Some(b',') => {
                *cursor += 1;
            }
            Some(b']') => {
                *cursor += 1;
                return Ok(Json::Array(values));
            }
            _ => return Err(format!("expected ',' or ']' at byte {cursor}")),
        }
    }
}

fn parse_string(bytes: &[u8], cursor: &mut usize) -> Result<String, String> {
    expect(bytes, cursor, b'"')?;
    let mut out = String::new();
    loop {
        let byte = match bytes.get(*cursor) {
            Some(byte) => *byte,
            None => return Err("unterminated string".to_string()),
        };
        *cursor += 1;
        match byte {
            b'"' => return Ok(out),
            b'\\' => {
                let escape = *bytes
                    .get(*cursor)
                    .ok_or_else(|| "unterminated escape".to_string())?;
                *cursor += 1;
                match escape {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000C}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let code = parse_hex4(bytes, cursor)?;
                        if (0xD800..0xDC00).contains(&code) && bytes.get(*cursor) == Some(&b'\\') {
                            *cursor += 1;
                            if bytes.get(*cursor) != Some(&b'u') {
                                return Err("malformed surrogate pair".to_string());
                            }
                            *cursor += 1;
                            let low = parse_hex4(bytes, cursor)?;
                            let combined =
                                0x10000 + ((code - 0xD800) << 10) + (low.wrapping_sub(0xDC00));
                            out.push(
                                char::from_u32(combined)
                                    .ok_or_else(|| "invalid code point".to_string())?,
                            );
                        } else {
                            out.push(
                                char::from_u32(code)
                                    .ok_or_else(|| "invalid code point".to_string())?,
                            );
                        }
                    }
                    _ => return Err(format!("invalid escape at byte {cursor}")),
                }
            }
            byte if byte < 0x20 => return Err("control character in string".to_string()),
            byte if byte < 0x80 => out.push(byte as char),
            _ => {
                // Multi-byte UTF-8: rewind and let `str` decode it.
                let start = *cursor - 1;
                let tail = std::str::from_utf8(&bytes[start..])
                    .map_err(|_| "invalid UTF-8 in string".to_string())?;
                let ch = tail.chars().next().ok_or_else(|| "bad UTF-8".to_string())?;
                out.push(ch);
                *cursor = start + ch.len_utf8();
            }
        }
    }
}

fn parse_hex4(bytes: &[u8], cursor: &mut usize) -> Result<u32, String> {
    let mut value = 0u32;
    for _ in 0..4 {
        let byte = *bytes
            .get(*cursor)
            .ok_or_else(|| "short unicode escape".to_string())?;
        *cursor += 1;
        let digit = match byte {
            b'0'..=b'9' => (byte - b'0') as u32,
            b'a'..=b'f' => (byte - b'a' + 10) as u32,
            b'A'..=b'F' => (byte - b'A' + 10) as u32,
            _ => return Err("invalid unicode escape".to_string()),
        };
        value = value * 16 + digit;
    }
    Ok(value)
}

fn parse_number(bytes: &[u8], cursor: &mut usize) -> Result<Json, String> {
    let start = *cursor;
    if bytes.get(*cursor) == Some(&b'-') {
        *cursor += 1;
    }
    while matches!(bytes.get(*cursor), Some(b'0'..=b'9')) {
        *cursor += 1;
    }
    if bytes.get(*cursor) == Some(&b'.') {
        *cursor += 1;
        while matches!(bytes.get(*cursor), Some(b'0'..=b'9')) {
            *cursor += 1;
        }
    }
    if matches!(bytes.get(*cursor), Some(b'e' | b'E')) {
        *cursor += 1;
        if matches!(bytes.get(*cursor), Some(b'+' | b'-')) {
            *cursor += 1;
        }
        while matches!(bytes.get(*cursor), Some(b'0'..=b'9')) {
            *cursor += 1;
        }
    }
    if *cursor == start {
        return Err(format!("invalid value at byte {start}"));
    }
    let text = std::str::from_utf8(&bytes[start..*cursor])
        .map_err(|_| "invalid number bytes".to_string())?;
    text.parse::<f64>()
        .map(Json::Number)
        .map_err(|_| format!("invalid number '{text}'"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_nested_values() {
        let value = Json::object(vec![
            ("name", Json::string("earth \"frame\"\n")),
            ("values", Json::numbers(&[1.0, -2.5, 1440.0])),
            ("flags", Json::array(vec![Json::Bool(true), Json::Null])),
        ]);
        let text = value.to_string();
        let parsed = Json::parse(&text).expect("parse");
        assert_eq!(parsed, value);
    }

    #[test]
    fn rejects_trailing_and_malformed_input() {
        assert!(Json::parse("{}x").is_err());
        assert!(Json::parse("[1,]").is_err());
        assert!(Json::parse("{\"a\":}").is_err());
        assert!(Json::parse("").is_err());
    }

    #[test]
    fn integral_numbers_have_no_trailing_zero() {
        assert_eq!(Json::Number(1440.0).to_string(), "1440");
        assert_eq!(Json::Number(0.5).to_string(), "0.5");
    }
}
