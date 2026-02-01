#pragma once

#include <string>
#include <memory>

/**
 * Fingerprint Manager - Manages fingerprint authentication
 * 
 * This is a placeholder wrapper that will integrate with libfprint
 * when available. For now, it's a stub to match the face verification interface.
 */
class FingerprintManager {
public:
    static FingerprintManager& instance();
    
    /**
     * Initialize fingerprint subsystem
     */
    bool initialize();
    
    /**
     * Check if fingerprint reader is available
     */
    bool is_available() const { return reader_available_; }
    
    /**
     * Verify fingerprint for the given username
     * Returns confidence score (0.0 to 1.0), or -1.0 on error
     */
    float verify_fingerprint(const std::string& username);
    
    /**
     * Get status string for logging
     */
    std::string get_status() const;
    
    ~FingerprintManager();
    
private:
    FingerprintManager() = default;
    
    bool reader_available_ = false;
    // TODO: Add libfprint device object when integrating
};
