#include "logger.h"
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>
#include <syslog.h>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <vector>
#include <grp.h>
#include <fcntl.h>

namespace faceid {

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

Logger::Logger() : write_buffer_(0), buffer_position_(0) {
    // Pre-allocate circular buffer to avoid reallocations
    write_buffer_.reserve(BUFFER_SIZE);
    
    // Skip log file opening in PAM context to avoid stderr warnings
    // that break pkttyagent (polkit) authentication
    const char* pam_context = std::getenv("FACEID_PAM_CONTEXT");
    if (pam_context == nullptr) {
        // Not in PAM context - open log file normally
        setLogFile("/var/log/faceid.log");
    }
    // In PAM context: skip setLogFile(), use syslog only
}

Logger::~Logger() {
    // Flush any remaining buffered data before closing
    flush();
    
    if (log_file_.is_open()) {
        log_file_.close();
    }
}

void Logger::setLogFile(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (log_file_.is_open()) {
        log_file_.close();
    }
    
    log_file_path_ = path;
    
    // Check if file exists
    bool file_exists = (access(path.c_str(), F_OK) == 0);
    
    // If file doesn't exist, create it with proper permissions
    if (!file_exists) {
        // Create file with 664 permissions (rw-rw-r--)
        int fd = open(path.c_str(), O_CREAT | O_WRONLY | O_APPEND, 0664);
        if (fd >= 0) {
            // Try to set group to 'log' (gid 19) for shared access
            struct group *grp = getgrnam("log");
            if (grp != nullptr) {
                fchown(fd, -1, grp->gr_gid);  // Keep owner, change group
            }
            close(fd);
        }
    }
    
    log_file_.open(path, std::ios::app);
    if (!log_file_.is_open()) {
        // Fallback to stderr
        console_output_ = true;
        std::cerr << "Warning: Could not open log file " << path 
                  << ", falling back to console output" << std::endl;
    }
}

void Logger::setLogLevel(LogLevel level) {
    min_level_ = level;
}

std::string Logger::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    
    return ss.str();
}

std::string Logger::levelToString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR:   return "ERROR";
        default:                return "UNKNOWN";
    }
}

void Logger::rotateLogIfNeeded() {
    // Only perform rotation check periodically (every 50 writes to reduce overhead)
    if (log_counter_ < 50) {
        log_counter_++;
        return;
    }
    log_counter_ = 0;
    
    if (log_file_path_.empty() || console_output_) {
        return;
    }
    
    // STREAMING APPROACH: Count lines without reading entire file into memory
    // Phase 1: Count total lines by scanning file
    std::ifstream infile(log_file_path_);
    if (!infile.is_open()) {
        return;
    }
    
    size_t line_count = 0;
    std::string line;
    line.reserve(256);  // Pre-allocate to reduce reallocations
    
    while (std::getline(infile, line)) {
        line_count++;
    }
    infile.close();
    
    // If we have more than max_log_lines_, keep only the last max_log_lines_
    if (line_count > max_log_lines_) {
        size_t lines_to_skip = line_count - max_log_lines_;
        
        // Phase 2: Copy only the last max_log_lines_ to a temp file
        std::ifstream infile2(log_file_path_);
        if (!infile2.is_open()) {
            return;
        }
        
        std::string temp_path = log_file_path_ + ".tmp";
        std::ofstream outfile(temp_path, std::ios::trunc);
        if (!outfile.is_open()) {
            infile2.close();
            return;
        }
        
        // Skip old lines
        size_t current_line = 0;
        while (current_line < lines_to_skip && std::getline(infile2, line)) {
            current_line++;
        }
        
        // Copy remaining lines (streaming, no vector allocation)
        while (std::getline(infile2, line)) {
            outfile << line << '\n';
        }
        
        infile2.close();
        outfile.close();
        
        // Close the current log file before replacing
        if (log_file_.is_open()) {
            log_file_.close();
        }
        
        // Replace old file with rotated file
        if (rename(temp_path.c_str(), log_file_path_.c_str()) != 0) {
            // Rename failed, try manual copy+delete as fallback
            std::ifstream src(temp_path, std::ios::binary);
            std::ofstream dst(log_file_path_, std::ios::binary | std::ios::trunc);
            dst << src.rdbuf();
            src.close();
            dst.close();
            unlink(temp_path.c_str());
        }
        
        // Reopen the file in append mode
        log_file_.open(log_file_path_, std::ios::app);
    }
}

void Logger::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Write buffered data to file
    if (buffer_position_ > 0 && log_file_.is_open()) {
        log_file_.write(write_buffer_.data(), buffer_position_);
        log_file_.flush();
        buffer_position_ = 0;
        write_buffer_.clear();
    }
}

void Logger::log(LogLevel level, const std::string& message) {
    if (level < min_level_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::stringstream ss;
    ss << "[" << getCurrentTimestamp() << "] "
       << "[" << levelToString(level) << "] "
       << "[PID:" << getpid() << "] "
       << message << std::endl;
    
    std::string log_line = ss.str();
    
    if (console_output_) {
        std::cerr << log_line;
    } else if (log_file_.is_open()) {
        // Use circular buffer for batching (Fix #6)
        // Add log line to buffer
        const char* data = log_line.c_str();
        size_t len = log_line.length();
        
        // Check if buffer will overflow
        if (buffer_position_ + len >= BUFFER_SIZE) {
            // Buffer is full, flush immediately
            if (buffer_position_ > 0) {
                log_file_.write(write_buffer_.data(), buffer_position_);
                log_file_.flush();
                buffer_position_ = 0;
                write_buffer_.clear();
            }
        }
        
        // Add to buffer
        write_buffer_.insert(write_buffer_.end(), data, data + len);
        buffer_position_ += len;
        
        // Flush if buffer reaches threshold (4KB)
        if (buffer_position_ >= FLUSH_THRESHOLD) {
            log_file_.write(write_buffer_.data(), buffer_position_);
            log_file_.flush();
            buffer_position_ = 0;
            write_buffer_.clear();
        }
    } else {
        // If no file and no console (PAM context), use syslog as fallback
        int syslog_level = LOG_INFO;
        switch (level) {
            case LogLevel::DEBUG:   syslog_level = LOG_DEBUG; break;
            case LogLevel::INFO:    syslog_level = LOG_INFO; break;
            case LogLevel::WARNING: syslog_level = LOG_WARNING; break;
            case LogLevel::ERROR:   syslog_level = LOG_ERR; break;
        }
        syslog(syslog_level, "%s", message.c_str());
    }
    
    // Check if rotation is needed (every 50 writes)
    rotateLogIfNeeded();
}

void Logger::debug(const std::string& message) {
    log(LogLevel::DEBUG, message);
}

void Logger::info(const std::string& message) {
    log(LogLevel::INFO, message);
}

void Logger::warning(const std::string& message) {
    log(LogLevel::WARNING, message);
}

void Logger::error(const std::string& message) {
    log(LogLevel::ERROR, message);
}

void Logger::auditAuthAttempt(const std::string& username, const std::string& method) {
    std::stringstream ss;
    ss << "AUTH_ATTEMPT user=" << username 
       << " method=" << method;
    info(ss.str());
}

void Logger::auditAuthSuccess(const std::string& username, const std::string& method, double duration_ms) {
    std::stringstream ss;
    ss << "AUTH_SUCCESS user=" << username 
       << " method=" << method 
       << " duration=" << std::fixed << std::setprecision(2) << duration_ms << "ms";
    info(ss.str());
}

void Logger::auditAuthFailure(const std::string& username, const std::string& method, const std::string& reason) {
    std::stringstream ss;
    ss << "AUTH_FAILURE user=" << username 
       << " method=" << method 
       << " reason=" << reason;
    warning(ss.str());
}

} // namespace faceid
