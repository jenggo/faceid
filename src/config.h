#ifndef FACEID_CONFIG_H
#define FACEID_CONFIG_H

#include <string>
#include <map>
#include <optional>

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

private:
    Config() = default;
    std::map<std::string, std::map<std::string, std::string>> data_;
    
    std::string trim(const std::string& str) const;
};

} // namespace faceid

#endif // FACEID_CONFIG_H
