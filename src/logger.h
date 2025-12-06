#ifndef FACEID_LOGGER_H
#define FACEID_LOGGER_H

#include <string>
#include <fstream>
#include <mutex>
#include <cstddef>

namespace faceid {

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

class Logger {
public:
    static Logger& getInstance();
    
    void setLogFile(const std::string& path);
    void setLogLevel(LogLevel level);
    
    void debug(const std::string& message);
    void info(const std::string& message);
    void warning(const std::string& message);
    void error(const std::string& message);
    
    // Audit trail specific methods
    void auditAuthAttempt(const std::string& username, const std::string& method);
    void auditAuthSuccess(const std::string& username, const std::string& method, double duration_ms);
    void auditAuthFailure(const std::string& username, const std::string& method, const std::string& reason);

private:
    Logger();
    ~Logger();
    
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    void log(LogLevel level, const std::string& message);
    std::string getCurrentTimestamp();
    std::string levelToString(LogLevel level);
    void rotateLogIfNeeded();
    
    std::ofstream log_file_;
    std::mutex mutex_;
    LogLevel min_level_ = LogLevel::INFO;
    bool console_output_ = false;
    std::string log_file_path_;
    size_t max_log_lines_ = 50;
    size_t log_counter_ = 0;
};

} // namespace faceid

#endif // FACEID_LOGGER_H
