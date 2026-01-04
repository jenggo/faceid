#ifndef FACEID_LOGGER_H
#define FACEID_LOGGER_H

#include <string>
#include <fstream>
#include <mutex>
#include <cstddef>
#include <vector>

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
    
    // Flush buffer to disk (called manually or on buffer full)
    void flush();

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
    size_t max_log_lines_ = 1000;  // Rotate after 1000 lines (reduced memory usage)
    size_t log_counter_ = 0;
    
    // Circular buffer for batching writes (Fix #6)
    static const size_t BUFFER_SIZE = 8 * 1024 * 1024;  // 8 MB buffer
    std::vector<char> write_buffer_;
    size_t buffer_position_ = 0;
    static const size_t FLUSH_THRESHOLD = 4096;  // Flush after 4KB
};

} // namespace faceid

#endif // FACEID_LOGGER_H
