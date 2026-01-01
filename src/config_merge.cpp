/**
 * Config Merge Utility
 * Merges configuration files while preserving user values
 * 
 * Usage: faceid-config-merge <source_config> <dest_config>
 * 
 * Strategy:
 * 1. Read all user values from existing dest_config
 * 2. Read structure and new keys from source_config
 * 3. Write merged config: source structure + user values
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include <regex>
#include <ctime>

struct ConfigLine {
    enum Type { COMMENT, SECTION, KEYVALUE, EMPTY };
    Type type;
    std::string content;        // Full line for comment/empty
    std::string section;        // For section headers
    std::string key;            // For key-value pairs
    std::string value;          // For key-value pairs
    std::string indent;         // Preserve indentation
};

std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

ConfigLine parseLine(const std::string& line) {
    ConfigLine result;
    
    // Empty line
    if (trim(line).empty()) {
        result.type = ConfigLine::EMPTY;
        result.content = line;
        return result;
    }
    
    // Comment
    std::regex comment_regex(R"(^\s*#)");
    if (std::regex_search(line, comment_regex)) {
        result.type = ConfigLine::COMMENT;
        result.content = line;
        return result;
    }
    
    // Section header
    std::regex section_regex(R"(^\s*\[([^\]]+)\]\s*$)");
    std::smatch section_match;
    if (std::regex_search(line, section_match, section_regex)) {
        result.type = ConfigLine::SECTION;
        result.section = section_match[1];
        result.content = line;
        return result;
    }
    
    // Key-value pair
    std::regex kv_regex(R"(^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$)");
    std::smatch kv_match;
    if (std::regex_search(line, kv_match, kv_regex)) {
        result.type = ConfigLine::KEYVALUE;
        result.indent = kv_match[1];
        result.key = kv_match[2];
        result.value = trim(kv_match[3].str());
        return result;
    }
    
    // Unknown, treat as comment
    result.type = ConfigLine::COMMENT;
    result.content = line;
    return result;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <source_config> <dest_config>" << std::endl;
        return 1;
    }
    
    std::string source_path = argv[1];
    std::string dest_path = argv[2];
    
    // Check source exists
    std::ifstream source_check(source_path);
    if (!source_check.good()) {
        std::cerr << "Error: Source config not found: " << source_path << std::endl;
        return 1;
    }
    source_check.close();
    
    // Check if destination exists
    std::ifstream dest_check(dest_path);
    bool dest_exists = dest_check.good();
    dest_check.close();
    
    // If destination doesn't exist, just copy source
    if (!dest_exists) {
        std::ifstream source(source_path);
        std::ofstream dest(dest_path);
        dest << source.rdbuf();
        std::cout << "New config installed: " << dest_path << std::endl;
        return 0;
    }
    
    std::cout << "Merging configuration..." << std::endl;
    
    // Read existing user values from destination
    // Map: "section|key" -> value
    std::map<std::string, std::string> user_values;
    std::string current_section;
    
    std::ifstream dest_file(dest_path);
    std::string line;
    while (std::getline(dest_file, line)) {
        ConfigLine parsed = parseLine(line);
        
        if (parsed.type == ConfigLine::SECTION) {
            current_section = parsed.section;
        } else if (parsed.type == ConfigLine::KEYVALUE) {
            std::string lookup_key = current_section + "|" + parsed.key;
            user_values[lookup_key] = parsed.value;
        }
    }
    dest_file.close();
    
    std::cout << "Found " << user_values.size() << " existing user settings" << std::endl;
    
    // Build a set of valid keys from source config
    std::map<std::string, bool> valid_keys;
    current_section = "";
    
    std::ifstream source_scan(source_path);
    while (std::getline(source_scan, line)) {
        ConfigLine parsed = parseLine(line);
        
        if (parsed.type == ConfigLine::SECTION) {
            current_section = parsed.section;
        } else if (parsed.type == ConfigLine::KEYVALUE) {
            std::string lookup_key = current_section + "|" + parsed.key;
            valid_keys[lookup_key] = true;
        }
    }
    source_scan.close();
    
    // Detect obsolete keys (exist in user config but not in source)
    std::vector<std::string> obsolete_keys;
    for (const auto& kv : user_values) {
        if (valid_keys.find(kv.first) == valid_keys.end()) {
            obsolete_keys.push_back(kv.first);
        }
    }
    
    // First pass: Check if there are any changes (new keys or different values)
    bool has_changes = false;
    int potential_new_keys = 0;
    current_section = "";
    
    std::ifstream source_precheck(source_path);
    while (std::getline(source_precheck, line)) {
        ConfigLine parsed = parseLine(line);
        
        if (parsed.type == ConfigLine::SECTION) {
            current_section = parsed.section;
        } else if (parsed.type == ConfigLine::KEYVALUE) {
            std::string lookup_key = current_section + "|" + parsed.key;
            if (user_values.find(lookup_key) == user_values.end()) {
                has_changes = true;
                potential_new_keys++;
            }
        }
    }
    source_precheck.close();
    
    // Check if there are obsolete keys to remove
    if (!obsolete_keys.empty()) {
        has_changes = true;
    }
    
    // If no changes, just report and exit
    if (!has_changes) {
        std::cout << "No configuration changes detected - backup skipped" << std::endl;
        std::cout << "Config already up-to-date: " << dest_path << std::endl;
        return 0;
    }
    
    if (potential_new_keys > 0) {
        std::cout << "Detected " << potential_new_keys << " new configuration keys" << std::endl;
    }
    
    if (!obsolete_keys.empty()) {
        std::cout << "Detected " << obsolete_keys.size() << " obsolete configuration keys (will be removed)" << std::endl;
    }
    
    // Create backup only if there are changes
    std::time_t now = std::time(nullptr);
    char timestamp[32];
    std::strftime(timestamp, sizeof(timestamp), "%Y%m%d-%H%M%S", std::localtime(&now));
    std::string backup_path = dest_path + ".backup." + timestamp;
    
    std::ifstream backup_src(dest_path, std::ios::binary);
    std::ofstream backup_dst(backup_path, std::ios::binary);
    backup_dst << backup_src.rdbuf();
    backup_src.close();
    backup_dst.close();
    
    std::cout << "Backup created: " << backup_path << std::endl;
    
    // Process source config and merge
    std::ifstream source_file(source_path);
    std::ofstream merged_file(dest_path);
    
    current_section = "";
    int added_keys = 0;
    int preserved_keys = 0;
    std::vector<std::string> new_keys;
    
    while (std::getline(source_file, line)) {
        ConfigLine parsed = parseLine(line);
        
        if (parsed.type == ConfigLine::SECTION) {
            current_section = parsed.section;
            merged_file << line << std::endl;
            
        } else if (parsed.type == ConfigLine::KEYVALUE) {
            std::string lookup_key = current_section + "|" + parsed.key;
            
            auto it = user_values.find(lookup_key);
            if (it != user_values.end()) {
                // User has custom value, preserve it
                merged_file << parsed.indent << parsed.key << " = " << it->second << std::endl;
                preserved_keys++;
            } else {
                // New key, use default from source
                merged_file << line << std::endl;
                added_keys++;
                new_keys.push_back("  + " + parsed.indent + parsed.key + " = " + parsed.value);
            }
            
        } else {
            // Comment, empty line, or unknown - preserve as-is
            merged_file << line << std::endl;
        }
    }
    
    source_file.close();
    merged_file.close();
    
    std::cout << "Configuration merge complete:" << std::endl;
    std::cout << "  - Preserved user values: " << preserved_keys << std::endl;
    std::cout << "  - Added new keys: " << added_keys << std::endl;
    if (!obsolete_keys.empty()) {
        std::cout << "  - Removed obsolete keys: " << obsolete_keys.size() << std::endl;
    }
    std::cout << "Config updated: " << dest_path << std::endl;
    
    if (!new_keys.empty()) {
        std::cout << std::endl;
        std::cout << "New configuration options added:" << std::endl;
        for (const auto& key : new_keys) {
            std::cout << key << std::endl;
        }
    }
    
    if (!obsolete_keys.empty()) {
        std::cout << std::endl;
        std::cout << "Obsolete configuration options removed:" << std::endl;
        for (const auto& key : obsolete_keys) {
            // Parse section|key format
            size_t pipe_pos = key.find('|');
            std::string section = key.substr(0, pipe_pos);
            std::string keyname = key.substr(pipe_pos + 1);
            std::cout << "  - [" << section << "] " << keyname << std::endl;
        }
    }
    
    return 0;
}
