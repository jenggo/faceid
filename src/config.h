#ifndef FACEID_CONFIG_H
#define FACEID_CONFIG_H

#include <string>
#include <map>
#include <optional>
#include <vector>

namespace faceid {

class Config {
public:
    static Config& getInstance();
    
    bool load(const std::string& path);
    
    std::optional<std::string> getString(const std::string& section, const std::string& key) const;
    std::optional<int> getInt(const std::string& section, const std::string& key) const;
    std::optional<double> getDouble(const std::string& section, const std::string& key) const;
    std::optional<bool> getBool(const std::string& section, const std::string& key) const;
    
    void set(const std::string& section, const std::string& key, const std::string& value);
    bool save(const std::string& path);
    
    // Get validation errors from last load
    std::vector<std::string> getValidationErrors() const { return validation_errors_; }

private:
    Config() = default;
    std::map<std::string, std::map<std::string, std::string>> data_;
    std::vector<std::string> validation_errors_;
    
    std::string trim(const std::string& str) const;
    bool validate();
    bool validateInt(const std::string& section, const std::string& key, int min_val, int max_val);
    bool validateDouble(const std::string& section, const std::string& key, double min_val, double max_val);
};

} // namespace faceid

#endif // FACEID_CONFIG_H
