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

namespace faceid {

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

Logger::Logger() {
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
    // Only perform rotation check periodically (every 10 writes)
    if (log_counter_ < 10) {
        log_counter_++;
        return;
    }
    log_counter_ = 0;
    
    if (log_file_path_.empty() || console_output_) {
        return;
    }
    
    // Read all lines from the log file
    std::ifstream infile(log_file_path_);
    if (!infile.is_open()) {
        return;
    }
    
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(infile, line)) {
        lines.push_back(line);
    }
    infile.close();
    
    // If we have more than max_log_lines_, keep only the last max_log_lines_
    if (lines.size() > max_log_lines_) {
        size_t start_index = lines.size() - max_log_lines_;
        
        // Close the file before rewriting
        if (log_file_.is_open()) {
            log_file_.close();
        }
        
        // Rewrite with only the last N lines
        std::ofstream outfile(log_file_path_, std::ios::trunc);
        if (outfile.is_open()) {
            for (size_t i = start_index; i < lines.size(); ++i) {
                outfile << lines[i] << std::endl;
            }
            outfile.close();
        }
        
        // Reopen the file in append mode
        log_file_.open(log_file_path_, std::ios::app);
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
    
    if (console_output_) {
        std::cerr << ss.str();
    } else if (log_file_.is_open()) {
        log_file_ << ss.str();
        log_file_.flush();
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
    
    // Check if rotation is needed (every 10 writes)
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
