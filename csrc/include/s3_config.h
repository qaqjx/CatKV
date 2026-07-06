#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>

namespace s3 {

struct S3Config {
  std::string endpoint;
  std::string access_key;
  std::string secret_key;
  std::string bucket;
  std::string region;
  bool use_ssl{false};
};

// Example config:
// [s3]
// endpoint = localhost:9000
// access_key = minioadmin
// secret_key = minioadmin
// bucket = kv-cache
// region = us-east-1
// use_ssl = false

inline std::string trim(const std::string &value) {
  const auto begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos)
    return {};
  const auto end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

inline std::optional<bool> parse_bool(const std::string &value) {
  std::string lower = value;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (lower == "true" || lower == "1")
    return true;
  if (lower == "false" || lower == "0")
    return false;
  return std::nullopt;
}

inline bool load_s3_config(const std::string &path, S3Config &config) {
  std::ifstream file(path);
  if (!file.is_open())
    return false;

  bool in_s3_section = false;
  std::string line;
  while (std::getline(file, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#')
      continue;

    if (line.front() == '[' && line.back() == ']') {
      in_s3_section = (trim(line.substr(1, line.size() - 2)) == "s3");
      continue;
    }

    if (!in_s3_section)
      continue;

    const auto equal_pos = line.find('=');
    if (equal_pos == std::string::npos)
      continue;

    const std::string key = trim(line.substr(0, equal_pos));
    const std::string value = trim(line.substr(equal_pos + 1));

    if (key == "endpoint")
      config.endpoint = value;
    else if (key == "access_key")
      config.access_key = value;
    else if (key == "secret_key")
      config.secret_key = value;
    else if (key == "bucket")
      config.bucket = value;
    else if (key == "region")
      config.region = value;
    else if (key == "use_ssl") {
      const auto parsed = parse_bool(value);
      if (parsed.has_value())
        config.use_ssl = parsed.value();
    }
  }

  return !config.endpoint.empty() && !config.access_key.empty() &&
         !config.secret_key.empty() && !config.bucket.empty() &&
         !config.region.empty();
}

} // namespace s3

