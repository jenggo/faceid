#include "config.h"
#include <fstream>
#include <sstream>
#include <algorithm>

namespace faceid {

Config& Config::getInstance() {
    static Config instance;
    return instance;
}

std::string Config::trim(const std::string& str) const {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

bool Config::load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    std::string current_section;

    while (std::getline(file, line)) {
        line = trim(line);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#' || line[0] == ';') {
            continue;
        }

        // Section header
        if (line[0] == '[' && line.back() == ']') {
            current_section = line.substr(1, line.length() - 2);
            continue;
        }

        // Key-value pair
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = trim(line.substr(0, pos));
            std::string value = trim(line.substr(pos + 1));
            data_[current_section][key] = value;
        }
    }

    return true;
}

std::optional<std::string> Config::getString(const std::string& section, const std::string& key) const {
    auto section_it = data_.find(section);
    if (section_it == data_.end()) {
        return std::nullopt;
    }

    auto key_it = section_it->second.find(key);
    if (key_it == section_it->second.end()) {
        return std::nullopt;
    }

    return key_it->second;
}

std::optional<int> Config::getInt(const std::string& section, const std::string& key) const {
    auto value = getString(section, key);
    if (!value) return std::nullopt;
    
    try {
        return std::stoi(*value);
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<double> Config::getDouble(const std::string& section, const std::string& key) const {
    auto value = getString(section, key);
    if (!value) return std::nullopt;
    
    try {
        return std::stod(*value);
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<bool> Config::getBool(const std::string& section, const std::string& key) const {
    auto value = getString(section, key);
    if (!value) return std::nullopt;
    
    std::string lower = *value;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "true" || lower == "yes" || lower == "1" || lower == "on") {
        return true;
    } else if (lower == "false" || lower == "no" || lower == "0" || lower == "off") {
        return false;
    }
    
    return std::nullopt;
}

void Config::set(const std::string& section, const std::string& key, const std::string& value) {
    data_[section][key] = value;
}

bool Config::save(const std::string& path) {
    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }

    for (const auto& [section, keys] : data_) {
        file << "[" << section << "]\n";
        for (const auto& [key, value] : keys) {
            file << key << " = " << value << "\n";
        }
        file << "\n";
    }

    return true;
}

} // namespace faceid
