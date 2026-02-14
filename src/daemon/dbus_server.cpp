#include "dbus_server.h"
#include "auth_manager.h"
#include "enroll_manager.h"
#include "fingerprint_manager.h"
#include "face_detector_manager.h"
#include "../image.h"
#include "../path_utils.h"
#include <pwd.h>
#include <grp.h>
#include <syslog.h>
#include <cstring>
#include <cerrno>
#include <fstream>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <nlohmann/json.hpp>
// JPEG decoding (match CLI which currently sends JPEG frames)
#include <turbojpeg.h>

// ============================================================================
// Helper Functions for Frame Processing
// ============================================================================

/**
 * Decode JPEG bytes to faceid::Image
 * Uses libturbojpeg for fast decompression
 * Returns empty Image on failure
 */
static faceid::Image decode_jpeg_frame(const uint8_t* jpeg_data, size_t jpeg_size) {
    if (!jpeg_data || jpeg_size < 4) {
        syslog(LOG_ERR, "Image decode: invalid data size %zu", jpeg_size);
        return faceid::Image();
    }

    // Detect PNG signature (explicitly reject since we expect JPEG)
    const uint8_t png_sig[] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    if (jpeg_size >= 8 && std::memcmp(jpeg_data, png_sig, 8) == 0) {
        syslog(LOG_ERR, "Image decode: received PNG data but daemon expects JPEG (libturbojpeg)");
        return faceid::Image();
    }

    // Detect JPEG by SOI marker 0xFF 0xD8
    if (jpeg_data[0] == 0xFF && jpeg_data[1] == 0xD8) {
        // Decode JPEG using libturbojpeg to match CLI encoder (which uses libturbojpeg)
        tjhandle tj = tjInitDecompress();
        if (!tj) {
            syslog(LOG_ERR, "JPEG decode: tjInitDecompress failed");
            return faceid::Image();
        }

        int width = 0, height = 0, jpeg_subsamp = 0, jpeg_colorspace = 0;
        int r = tjDecompressHeader3(tj, (const unsigned char*)jpeg_data, (unsigned long)jpeg_size,
                                    &width, &height, &jpeg_subsamp, &jpeg_colorspace);
        if (r < 0) {
            syslog(LOG_ERR, "JPEG decode: header parse failed: %s", tjGetErrorStr2(tj));
            tjDestroy(tj);
            return faceid::Image();
        }

        try {
            // Create Image with 3 channels (BGR) - CLI encodes BGR
            faceid::Image img(width, height, 3);

            // Row stride (bytes per row)
            int pitch = img.stride();

            // Decompress into BGR format
            r = tjDecompress2(tj,
                              (const unsigned char*)jpeg_data,
                              (unsigned long)jpeg_size,
                              img.data(),
                              width,
                              pitch,
                              height,
                              TJPF_BGR,
                              TJFLAG_FASTDCT);

            if (r < 0) {
                syslog(LOG_ERR, "JPEG decode: decompression failed: %s", tjGetErrorStr2(tj));
                tjDestroy(tj);
                return faceid::Image();
            }

            tjDestroy(tj);
            return img;
        } catch (const std::exception& e) {
            syslog(LOG_ERR, "JPEG decode: exception allocating image: %s", e.what());
            tjDestroy(tj);
            return faceid::Image();
        }
    }

    syslog(LOG_ERR, "Image decode: unsupported image format (expecting JPEG)");
    return faceid::Image();
}

// ============================================================================
// D-Bus Method Handlers - These are called by the D-Bus framework
// ============================================================================

// Global pointer to auth manager for use in C-style handlers
static AuthManager* g_auth_manager = nullptr;

/**
 * VerifyStart method handler
 * Called when PAM client calls org.freedesktop.FaceID.Device.VerifyStart
 * NoReply method - returns immediately, signals emit asynchronously
 */
static int method_verify_start(sd_bus_message* m, void* userdata, sd_bus_error* ret_error) {
    const char* username = nullptr;
    const char* context = nullptr;
    
    int r = sd_bus_message_read(m, "ss", &username, &context);
    if (r < 0) {
        sd_bus_error_set_const(ret_error, "org.freedesktop.DBus.Error.InvalidArgs", 
                              "Invalid arguments");
        return r;
    }
    
    DBusServer* server = static_cast<DBusServer*>(userdata);
    AuthManager* auth_mgr = server->get_auth_manager();
    
    if (!auth_mgr->verify_start(username, context)) {
        sd_bus_error_set_const(ret_error, "org.freedesktop.FaceID.Error.AlreadyVerifying",
                              "Authentication already in progress");
        return -EBUSY;
    }
    
    // NoReply method - don't send reply, async signals will follow
    return 0;
}

/**
 * VerifyStop method handler
 * Called when PAM client calls org.freedesktop.FaceID.Device.VerifyStop
 * NoReply method - cancels ongoing authentication
 */
static int method_verify_stop(sd_bus_message* m, void* userdata, sd_bus_error* ret_error) {
    DBusServer* server = static_cast<DBusServer*>(userdata);
    AuthManager* auth_mgr = server->get_auth_manager();
    
    auth_mgr->verify_stop();
    
    // NoReply method - don't send reply
    return 0;
}

/**
 * EnrollStart method handler
 * Called when GUI/CLI calls org.freedesktop.FaceID.Device.EnrollStart
 * Parameters: username (s), biometric_type (s, "face" or "fingerprint")
 * NoReply method - returns immediately, signals emit asynchronously
 */
static int method_enroll_start(sd_bus_message* m, void* userdata, sd_bus_error* ret_error) {
    const char* username = nullptr;
    const char* biometric_type = nullptr;
    
    int r = sd_bus_message_read(m, "ss", &username, &biometric_type);
    if (r < 0) {
        sd_bus_error_set_const(ret_error, "org.freedesktop.DBus.Error.InvalidArgs", 
                              "Invalid arguments");
        return r;
    }
    
    // Only support face enrollment for now
    if (std::string(biometric_type) != "face") {
        sd_bus_error_set_const(ret_error, "org.freedesktop.FaceID.Error.NotSupported",
                              "Only face biometric is currently supported");
        return -ENOTSUP;
    }
    
    DBusServer* server = static_cast<DBusServer*>(userdata);
    EnrollmentManager* enroll_mgr = server->get_enroll_manager();
    
    sd_bus* bus = sd_bus_message_get_bus(m);
    if (!enroll_mgr->enroll_start(bus, username)) {
        sd_bus_error_set_const(ret_error, "org.freedesktop.FaceID.Error.AlreadyEnrolling",
                              "Enrollment already in progress");
        return -EBUSY;
    }
    
    // NoReply method - don't send reply, async signals will follow
    return 0;
}

/**
 * GetStatus method handler
 * Returns daemon status: camera ready, models loaded, version
 */
static int method_get_status(sd_bus_message* m, void* userdata, sd_bus_error* ret_error) {
    // For now, return simple status dict
    // TODO: Replace with actual status from resource managers
    return sd_bus_reply_method_return(m, "a{sv}",
        6,  // 3 key-value pairs in the dict
        "version", "s", "2.0.0",
        "camera_ready", "b", 1,  // true
        "models_loaded", "i", 1
    );
}

/**
 * Helper: Emit FingerprintStatus signal
 * Parameters: status (string), progress (int), username (string), message (string)
 */
static void emit_fingerprint_status(sd_bus* bus, 
                                   const std::string& status,
                                   int progress,
                                   const std::string& username,
                                   const std::string& message) {
    int ret = sd_bus_emit_signal(
        bus,
        "/org/freedesktop/FaceID/Device",
        "org.freedesktop.FaceID.Device",
        "FingerprintStatus",
        "siss",
        status.c_str(),
        progress,
        username.c_str(),
        message.c_str()
    );
    
    if (ret < 0) {
        syslog(LOG_ERR, "Failed to emit FingerprintStatus signal: %s", strerror(-ret));
    }
}

/**
 * Helper: Emit FingerprintEnrollStatus signal
 */
static void emit_fingerprint_enroll_status(sd_bus* bus,
                                          const std::string& status,
                                          int progress,
                                          const std::string& username,
                                          const std::string& message) {
    int ret = sd_bus_emit_signal(
        bus,
        "/org/freedesktop/FaceID/Device",
        "org.freedesktop.FaceID.Device",
        "FingerprintEnrollStatus",
        "siss",
        status.c_str(),
        progress,
        username.c_str(),
        message.c_str()
    );
    
    if (ret < 0) {
        syslog(LOG_ERR, "Failed to emit FingerprintEnrollStatus signal: %s", strerror(-ret));
    }
}

/**
 * FingerprintVerify method handler
 * Verify user's fingerprint asynchronously
 */
static int method_fingerprint_verify(sd_bus_message* m, void* userdata, 
                                     sd_bus_error* ret_error) {
    // Parse parameters
    const char* username = nullptr;
    const char* context = nullptr;
    int ret = sd_bus_message_read(m, "ss", &username, &context);
    
    if (ret < 0) {
        syslog(LOG_ERR, "Failed to parse FingerprintVerify parameters: %s", strerror(-ret));
        return sd_bus_error_set_errno(ret_error, -ret);
    }
    
    // Validate username
    if (!username || strlen(username) == 0) {
        syslog(LOG_WARNING, "FingerprintVerify called with empty username");
        return sd_bus_error_set(ret_error, "org.freedesktop.FaceID.Error.InvalidArgument",
                               "Username cannot be empty");
    }
    
    // Get bus pointer
    sd_bus* bus = sd_bus_message_get_bus(m);
    std::string username_str(username);
    std::string context_str(context ? context : "");
    
    syslog(LOG_INFO, "FingerprintVerify requested for user: %s", username);
    
    // Launch verification in background thread (async pattern)
    std::thread([bus, username_str, context_str]() {
        faceid::daemon::FingerprintManager& fp_mgr = faceid::daemon::FingerprintManager::instance();
        std::atomic<bool> cancel_flag{false};
        
        // Check if fingerprint available
        if (!fp_mgr.is_available()) {
            emit_fingerprint_status(bus, "error", 0, username_str,
                                   "Fingerprint device not available");
            return;
        }
        
        // Emit in_progress signal
        emit_fingerprint_status(bus, "in_progress", 0, username_str,
                               "Place finger on scanner...");
        
        // Verify (blocking with timeout)
        float confidence = fp_mgr.verify_fingerprint(username_str, cancel_flag);
        
        // Emit result based on confidence
        if (confidence > 0.0f) {
            // Success - fingerprint matched
            emit_fingerprint_status(bus, "success", 100, username_str,
                                   "Fingerprint matched successfully");
        } else if (confidence == 0.0f) {
            // No match - fingerprint didn't match
            emit_fingerprint_status(bus, "no_match", 100, username_str,
                                   "Fingerprint did not match");
        } else {
            // Error or timeout
            faceid::FingerprintAuth& fp_auth = fp_mgr.get_fingerprint_auth();
            std::string error = fp_auth.getLastError();
            
            if (error.find("timeout") != std::string::npos) {
                emit_fingerprint_status(bus, "timeout", 100, username_str,
                                       "Fingerprint scan timeout");
            } else if (cancel_flag.load()) {
                emit_fingerprint_status(bus, "cancelled", 100, username_str,
                                       "Verification cancelled");
            } else {
                emit_fingerprint_status(bus, "error", 100, username_str, error);
            }
        }
    }).detach();  // Detach thread to avoid blocking D-Bus
    
    // Return immediately (async pattern)
    return sd_bus_reply_method_return(m, "");
}

/**
 * FingerprintEnroll method handler
 * Enroll user's fingerprint asynchronously
 */
static int method_fingerprint_enroll(sd_bus_message* m, void* userdata,
                                     sd_bus_error* ret_error) {
    // Parse parameters
    const char* username = nullptr;
    int ret = sd_bus_message_read(m, "s", &username);
    
    if (ret < 0) {
        syslog(LOG_ERR, "Failed to parse FingerprintEnroll parameters: %s", strerror(-ret));
        return sd_bus_error_set_errno(ret_error, -ret);
    }
    
    // Validate username
    if (!username || strlen(username) == 0) {
        syslog(LOG_WARNING, "FingerprintEnroll called with empty username");
        return sd_bus_error_set(ret_error, "org.freedesktop.FaceID.Error.InvalidArgument",
                               "Username cannot be empty");
    }
    
    // Get bus pointer
    sd_bus* bus = sd_bus_message_get_bus(m);
    std::string username_str(username);
    
    syslog(LOG_INFO, "FingerprintEnroll requested for user: %s", username);
    
    // Launch enrollment in background thread
    std::thread([bus, username_str]() {
        faceid::daemon::FingerprintManager& fp_mgr = faceid::daemon::FingerprintManager::instance();
        
        // Check if fingerprint reader available
        if (!fp_mgr.is_available()) {
            syslog(LOG_WARNING, "Fingerprint enrollment requested but device unavailable");
            emit_fingerprint_enroll_status(bus, "error", 0, username_str,
                                          "Fingerprint device not available");
            return;
        }
        
        syslog(LOG_INFO, "Starting fingerprint enrollment for user: %s", username_str.c_str());
        
        // Emit initial progress
        emit_fingerprint_enroll_status(bus, "in_progress", 0, username_str,
                                      "Place finger on scanner (sample 1/5)");
        
        // Get FingerprintAuth reference
        faceid::FingerprintAuth& fp_auth = fp_mgr.get_fingerprint_auth();
        std::atomic<bool> cancel_flag{false};
        
        // For now, use authenticate with special timeout to simulate enrollment
        // In a real implementation, we'd call fprintd enrollment API directly
        bool success = fp_auth.authenticate(username_str, 30, cancel_flag);
        
        if (cancel_flag.load()) {
            syslog(LOG_INFO, "Fingerprint enrollment cancelled for user: %s", username_str.c_str());
            emit_fingerprint_enroll_status(bus, "cancelled", 100, username_str,
                                          "Enrollment cancelled");
            return;
        }
        
        if (success) {
            syslog(LOG_INFO, "Fingerprint enrollment succeeded for user: %s", username_str.c_str());
            emit_fingerprint_enroll_status(bus, "success", 100, username_str,
                                          "Fingerprint enrolled successfully");
        } else {
            std::string error = fp_auth.getLastError();
            syslog(LOG_WARNING, "Fingerprint enrollment failed for user %s: %s",
                   username_str.c_str(), error.c_str());
            emit_fingerprint_enroll_status(bus, "error", 100, username_str, error);
        }
    }).detach();  // Detach thread
    
    // Return immediately (async pattern)
    return sd_bus_reply_method_return(m, "");
}

// ============================================================================
// Frame-Based ML Methods (for CLI with preview)
// ============================================================================

static int method_detect_face(sd_bus_message* m, void* userdata, sd_bus_error* ret_error) {
    // DetectFace(frame: byte[]) → (bounding_boxes: (iiii)[], quality: double)
    const uint8_t* jpeg_data = nullptr;
    size_t jpeg_size = 0;
    
    int r = sd_bus_message_read_array(m, 'y', (const void**)&jpeg_data, &jpeg_size);
    if (r < 0) {
        sd_bus_error_set_const(ret_error, "org.freedesktop.DBus.Error.InvalidArgs", 
                              "Invalid frame data");
        return r;
    }
    
    syslog(LOG_DEBUG, "DetectFace: received JPEG frame %zu bytes", jpeg_size);
    
    // Decode JPEG frame
    faceid::Image frame = decode_jpeg_frame(jpeg_data, jpeg_size);
    if (frame.empty()) {
        sd_bus_error_set_const(ret_error, "org.freedesktop.FaceID.Error.InvalidFrame",
                              "Failed to decode JPEG frame");
        return -EINVAL;
    }
    
    // Get face detector manager
    FaceDetectorManager& detector_mgr = FaceDetectorManager::instance();
    if (!detector_mgr.is_ready()) {
        sd_bus_error_set_const(ret_error, "org.freedesktop.FaceID.Error.NotReady",
                              "Face detector models not loaded");
        return -EAGAIN;
    }
    
    // Detect faces in frame
    std::vector<faceid::Rect> detected_faces = detector_mgr.detect_faces(frame);
    
    // Calculate quality metric (for now, simple heuristic based on face size)
    double quality = 0.0;
    if (!detected_faces.empty()) {
        // Use largest detected face for quality
        const faceid::Rect& largest = detected_faces[0];
        int face_area = largest.width * largest.height;
        int frame_area = frame.rows() * frame.cols();
        quality = (double)face_area / frame_area;  // Normalize by frame area
        // Cap quality at 1.0
        if (quality > 1.0) quality = 1.0;
    }
    
    syslog(LOG_DEBUG, "DetectFace: found %zu faces, quality=%.2f", detected_faces.size(), quality);
    
    // Build D-Bus response with array of bounding boxes
    // Signature: (iiii)d - array of structs (x, y, width, height), then double quality
    sd_bus_message* reply = nullptr;
    r = sd_bus_message_new_method_return(m, &reply);
    if (r < 0) {
        return r;
    }
    
    // Open array container for bounding boxes
    r = sd_bus_message_open_container(reply, 'a', "(iiii)");
    if (r < 0) {
        sd_bus_message_unref(reply);
        return r;
    }
    
    // Add each detected face box
    for (const auto& face : detected_faces) {
        r = sd_bus_message_append(reply, "(iiii)", 
            face.x, face.y, face.width, face.height);
        if (r < 0) {
            sd_bus_message_unref(reply);
            return r;
        }
    }
    
    // Close array container
    r = sd_bus_message_close_container(reply);
    if (r < 0) {
        sd_bus_message_unref(reply);
        return r;
    }
    
    // Add quality double
    r = sd_bus_message_append(reply, "d", quality);
    if (r < 0) {
        sd_bus_message_unref(reply);
        return r;
    }
    
    // Send reply
    r = sd_bus_send(nullptr, reply, nullptr);
    sd_bus_message_unref(reply);
    return r;
}

static int method_generate_embedding(sd_bus_message* m, void* userdata, sd_bus_error* ret_error) {
    // GenerateEmbedding(frame: byte[], bbox: (iiii)) → (embedding: byte[])
    const uint8_t* jpeg_data = nullptr;
    size_t jpeg_size = 0;
    int bbox_x = 0, bbox_y = 0, bbox_w = 0, bbox_h = 0;
    
    int r = sd_bus_message_read_array(m, 'y', (const void**)&jpeg_data, &jpeg_size);
    if (r < 0) {
        sd_bus_error_set_const(ret_error, "org.freedesktop.DBus.Error.InvalidArgs",
                              "Invalid frame data");
        return r;
    }
    
    r = sd_bus_message_read(m, "(iiii)", &bbox_x, &bbox_y, &bbox_w, &bbox_h);
    if (r < 0) {
        sd_bus_error_set_const(ret_error, "org.freedesktop.DBus.Error.InvalidArgs",
                              "Invalid bounding box");
        return r;
    }
    
    syslog(LOG_DEBUG, "GenerateEmbedding: frame %zu bytes, bbox (%d, %d, %d, %d)",
           jpeg_size, bbox_x, bbox_y, bbox_w, bbox_h);
    
    // Decode JPEG frame
    faceid::Image frame = decode_jpeg_frame(jpeg_data, jpeg_size);
    if (frame.empty()) {
        sd_bus_error_set_const(ret_error, "org.freedesktop.FaceID.Error.InvalidFrame",
                              "Failed to decode JPEG frame");
        return -EINVAL;
    }
    
    // Get face detector manager
    FaceDetectorManager& detector_mgr = FaceDetectorManager::instance();
    if (!detector_mgr.is_ready()) {
        sd_bus_error_set_const(ret_error, "org.freedesktop.FaceID.Error.NotReady",
                              "Face detector models not loaded");
        return -EAGAIN;
    }
    
    // Create Rect from D-Bus coordinates
    faceid::Rect face_rect(bbox_x, bbox_y, bbox_w, bbox_h);
    
    // Generate embedding for the specified face region
    faceid::FaceEncoding embedding = detector_mgr.encode_face(frame, face_rect);
    if (embedding.empty()) {
        sd_bus_error_set_const(ret_error, "org.freedesktop.FaceID.Error.EncodeFailed",
                              "Failed to encode face");
        return -EIO;
    }
    
    syslog(LOG_DEBUG, "GenerateEmbedding: generated %zu-dimensional embedding", embedding.size());
    
    // Convert embedding (std::vector<float>) to bytes for D-Bus transmission
    // Signature: (ay) - array of bytes
    sd_bus_message* reply = nullptr;
    r = sd_bus_message_new_method_return(m, &reply);
    if (r < 0) {
        return r;
    }
    
    // Embedding is float array, convert to bytes (each float = 4 bytes)
    std::vector<uint8_t> embedding_bytes;
    embedding_bytes.reserve(embedding.size() * sizeof(float));
    for (float val : embedding) {
        uint8_t* ptr = (uint8_t*)&val;
        embedding_bytes.insert(embedding_bytes.end(), ptr, ptr + sizeof(float));
    }
    
    r = sd_bus_message_append_array(reply, 'y', embedding_bytes.data(), embedding_bytes.size());
    if (r < 0) {
        sd_bus_message_unref(reply);
        return r;
    }
    
    // Send reply
    r = sd_bus_send(nullptr, reply, nullptr);
    sd_bus_message_unref(reply);
    return r;
}

static int method_verify_face(sd_bus_message* m, void* userdata, sd_bus_error* ret_error) {
    // VerifyFace(username: string, frame: byte[]) → (match: bool, confidence: double)
    const char* username = nullptr;
    const uint8_t* jpeg_data = nullptr;
    size_t jpeg_size = 0;
    
    int r = sd_bus_message_read(m, "s", &username);
    if (r < 0) {
        sd_bus_error_set_const(ret_error, "org.freedesktop.DBus.Error.InvalidArgs",
                              "Invalid username");
        return r;
    }
    
    r = sd_bus_message_read_array(m, 'y', (const void**)&jpeg_data, &jpeg_size);
    if (r < 0) {
        sd_bus_error_set_const(ret_error, "org.freedesktop.DBus.Error.InvalidArgs",
                              "Invalid frame data");
        return r;
    }
    
    syslog(LOG_DEBUG, "VerifyFace: user=%s, frame %zu bytes", username, jpeg_size);
    
    // Decode JPEG frame
    faceid::Image frame = decode_jpeg_frame(jpeg_data, jpeg_size);
    if (frame.empty()) {
        sd_bus_error_set_const(ret_error, "org.freedesktop.FaceID.Error.InvalidFrame",
                              "Failed to decode JPEG frame");
        return -EINVAL;
    }
    
    // Get face detector manager
    FaceDetectorManager& detector_mgr = FaceDetectorManager::instance();
    if (!detector_mgr.is_ready()) {
        sd_bus_error_set_const(ret_error, "org.freedesktop.FaceID.Error.NotReady",
                              "Face detector models not loaded");
        return -EAGAIN;
    }
    
    // Detect faces in the frame
    std::vector<faceid::Rect> detected_faces = detector_mgr.detect_faces(frame);
    if (detected_faces.empty()) {
        syslog(LOG_INFO, "VerifyFace: no faces detected for user %s", username);
        
        sd_bus_message* reply = nullptr;
        r = sd_bus_message_new_method_return(m, &reply);
        if (r < 0) return r;
        
        r = sd_bus_message_append(reply, "bd", false, 0.0);
        if (r < 0) {
            sd_bus_message_unref(reply);
            return r;
        }
        
        r = sd_bus_send(nullptr, reply, nullptr);
        sd_bus_message_unref(reply);
        return r;
    }
    
    // Load stored embeddings for the user
    // Path: ~/.local/share/faceid/faces/{username}.json
    std::string faces_dir = path_utils::get_user_faces_dir(username);
    if (faces_dir.empty()) {
        syslog(LOG_ERR, "VerifyFace: could not determine faces directory for user %s", username);
        sd_bus_error_set_const(ret_error, "org.freedesktop.FaceID.Error.SystemError",
                              "Could not determine user storage path");
        return -EIO;
    }
    std::string embedding_path = faces_dir + "/" + username + ".json";
    
    std::ifstream embedding_file(embedding_path);
    if (!embedding_file.is_open()) {
        syslog(LOG_WARNING, "VerifyFace: no enrollment for user %s", username);
        
        sd_bus_message* reply = nullptr;
        r = sd_bus_message_new_method_return(m, &reply);
        if (r < 0) return r;
        
        r = sd_bus_message_append(reply, "bd", false, 0.0);
        if (r < 0) {
            sd_bus_message_unref(reply);
            return r;
        }
        
        r = sd_bus_send(nullptr, reply, nullptr);
        sd_bus_message_unref(reply);
        return r;
    }
    
    // Parse stored embeddings (assuming JSON format)
    try {
        nlohmann::json enrollment_data;
        embedding_file >> enrollment_data;
        embedding_file.close();
        
        std::vector<faceid::FaceEncoding> stored_embeddings;
        
        // Assuming JSON structure: { "embeddings": [[...], [...], ...] }
        if (enrollment_data.contains("embeddings") && enrollment_data["embeddings"].is_array()) {
            for (const auto& emb_array : enrollment_data["embeddings"]) {
                faceid::FaceEncoding embedding;
                if (emb_array.is_array()) {
                    for (float val : emb_array) {
                        embedding.push_back(val);
                    }
                }
                if (!embedding.empty()) {
                    stored_embeddings.push_back(embedding);
                }
            }
        }
        
        if (stored_embeddings.empty()) {
            syslog(LOG_WARNING, "VerifyFace: no valid embeddings for user %s", username);
            
            sd_bus_message* reply = nullptr;
            r = sd_bus_message_new_method_return(m, &reply);
            if (r < 0) return r;
            
            r = sd_bus_message_append(reply, "bd", false, 0.0);
            if (r < 0) {
                sd_bus_message_unref(reply);
                return r;
            }
            
            r = sd_bus_send(nullptr, reply, nullptr);
            sd_bus_message_unref(reply);
            return r;
        }
        
        // Verify detected face against stored embeddings
        // Use the first (largest) detected face
        float best_confidence = detector_mgr.verify_face(frame, detected_faces, stored_embeddings);
        
        bool is_match = (best_confidence >= 0.5f);  // Confidence threshold
        
        syslog(LOG_INFO, "VerifyFace: user=%s, match=%d, confidence=%.2f",
               username, is_match, best_confidence);
        
        // Build response
        sd_bus_message* reply = nullptr;
        r = sd_bus_message_new_method_return(m, &reply);
        if (r < 0) return r;
        
        r = sd_bus_message_append(reply, "bd", is_match, (double)best_confidence);
        if (r < 0) {
            sd_bus_message_unref(reply);
            return r;
        }
        
        r = sd_bus_send(nullptr, reply, nullptr);
        sd_bus_message_unref(reply);
        return r;
        
    } catch (const std::exception& e) {
        syslog(LOG_ERR, "VerifyFace: JSON parse error for user %s: %s", username, e.what());
        sd_bus_error_set_const(ret_error, "org.freedesktop.FaceID.Error.ParseError",
                              "Failed to parse embedding data");
        return -EINVAL;
    }
}

static int method_save_embedding(sd_bus_message* m, void* userdata, sd_bus_error* ret_error) {
    // SaveEmbedding(username: string, embedding: byte[], pose: string) → ()
    const char* username = nullptr;
    const uint8_t* embedding_data = nullptr;
    size_t embedding_size = 0;
    const char* pose = nullptr;
    
    int r = sd_bus_message_read(m, "s", &username);
    if (r < 0) {
        sd_bus_error_set_const(ret_error, "org.freedesktop.DBus.Error.InvalidArgs",
                              "Invalid username");
        return r;
    }
    
    r = sd_bus_message_read_array(m, 'y', (const void**)&embedding_data, &embedding_size);
    if (r < 0) {
        sd_bus_error_set_const(ret_error, "org.freedesktop.DBus.Error.InvalidArgs",
                              "Invalid embedding data");
        return r;
    }
    
    r = sd_bus_message_read(m, "s", &pose);
    if (r < 0) {
        sd_bus_error_set_const(ret_error, "org.freedesktop.DBus.Error.InvalidArgs",
                              "Invalid pose");
        return r;
    }
    
    syslog(LOG_DEBUG, "SaveEmbedding: user=%s, embedding %zu bytes, pose=%s",
           username, embedding_size, pose);
    
    // Validate embedding size (should be 512 floats = 2048 bytes)
    if (embedding_size < 512 || embedding_size % 4 != 0) {
        sd_bus_error_set_const(ret_error, "org.freedesktop.FaceID.Error.InvalidEmbedding",
                              "Invalid embedding size");
        return -EINVAL;
    }
    
    // Convert bytes to float array
    std::vector<float> embedding;
    const float* float_ptr = (const float*)embedding_data;
    size_t float_count = embedding_size / sizeof(float);
    embedding.assign(float_ptr, float_ptr + float_count);
    
    // Ensure storage directory exists for user (and set ownership)
    std::string mkdir_err;
    if (!path_utils::ensure_faceid_directories(&mkdir_err)) {
        syslog(LOG_ERR, "SaveEmbedding: failed to ensure directory for %s: %s", username, mkdir_err.c_str());
        sd_bus_error_set_const(ret_error, "org.freedesktop.FaceID.Error.StorageError",
                              "Failed to create storage directory");
        return -EIO;
    }

    std::string faces_dir = path_utils::get_user_faces_dir(username);
    std::string embedding_path = faces_dir + "/" + username + ".json";
    
    // Load existing enrollments or create new
    nlohmann::json enrollment_data;
    std::ifstream input_file(embedding_path);
    if (input_file.is_open()) {
        try {
            input_file >> enrollment_data;
        } catch (const std::exception& e) {
            syslog(LOG_WARNING, "SaveEmbedding: failed to parse existing file for %s: %s",
                   username, e.what());
            enrollment_data = nlohmann::json();
        }
        input_file.close();
    }
    
    // Ensure embeddings array exists
    if (!enrollment_data.contains("embeddings")) {
        enrollment_data["embeddings"] = nlohmann::json::array();
    }
    
    // Add new embedding
    nlohmann::json emb_json = embedding;
    enrollment_data["embeddings"].push_back(emb_json);
    
    // Store metadata
    if (!enrollment_data.contains("poses")) {
        enrollment_data["poses"] = nlohmann::json::array();
    }
    enrollment_data["poses"].push_back(pose);
    
    // Write to file
    std::ofstream output_file(embedding_path);
    if (!output_file.is_open()) {
        syslog(LOG_ERR, "SaveEmbedding: failed to open file for writing: %s", strerror(errno));
        sd_bus_error_set_const(ret_error, "org.freedesktop.FaceID.Error.StorageError",
                              "Failed to write embedding file");
        return -EIO;
    }
    
    try {
        output_file << enrollment_data.dump(2);
        output_file.close();
    } catch (const std::exception& e) {
        syslog(LOG_ERR, "SaveEmbedding: failed to write file for %s: %s", username, e.what());
        sd_bus_error_set_const(ret_error, "org.freedesktop.FaceID.Error.StorageError",
                              "Failed to write embedding file");
        return -EIO;
    }
    
    // Set file permissions (0600 - only owner can read/write)
    if (chmod(embedding_path.c_str(), 0600) < 0) {
        syslog(LOG_WARNING, "SaveEmbedding: failed to set file permissions: %s", strerror(errno));
    }
    
    // Set file ownership to the target user
    struct passwd* pwd = getpwnam(username);
    if (pwd) {
        if (chown(embedding_path.c_str(), pwd->pw_uid, pwd->pw_gid) < 0) {
             syslog(LOG_WARNING, "SaveEmbedding: failed to chown file: %s", strerror(errno));
        }
    }

    syslog(LOG_INFO, "SaveEmbedding: saved embedding for user %s, pose %s",
           username, pose);
    
    // Send empty reply
    return sd_bus_reply_method_return(m, "");
}

static const sd_bus_vtable device_vtable[] = {
    SD_BUS_VTABLE_START(0),
    
    // Methods
    SD_BUS_METHOD("VerifyStart", "ss", "", method_verify_start, 
                  SD_BUS_VTABLE_UNPRIVILEGED),
    SD_BUS_METHOD("VerifyStop", "", "", method_verify_stop,
                  SD_BUS_VTABLE_UNPRIVILEGED),
    SD_BUS_METHOD("EnrollStart", "ss", "", method_enroll_start,
                  SD_BUS_VTABLE_UNPRIVILEGED),
    SD_BUS_METHOD("GetStatus", "", "a{sv}", method_get_status,
                  SD_BUS_VTABLE_UNPRIVILEGED),
    SD_BUS_METHOD("FingerprintVerify", "ss", "", method_fingerprint_verify,
                  SD_BUS_VTABLE_UNPRIVILEGED),
    SD_BUS_METHOD("FingerprintEnroll", "s", "", method_fingerprint_enroll,
                  SD_BUS_VTABLE_UNPRIVILEGED),
    
    // Frame-based ML processing methods (for CLI with preview)
    SD_BUS_METHOD("DetectFace", "ay", "a(iiii)d", method_detect_face,
                  SD_BUS_VTABLE_UNPRIVILEGED),
    SD_BUS_METHOD("GenerateEmbedding", "ay(iiii)", "ay", method_generate_embedding,
                  SD_BUS_VTABLE_UNPRIVILEGED),
    SD_BUS_METHOD("VerifyFace", "say", "bd", method_verify_face,
                  SD_BUS_VTABLE_UNPRIVILEGED),
    SD_BUS_METHOD("SaveEmbedding", "says", "", method_save_embedding,
                  SD_BUS_VTABLE_UNPRIVILEGED),
    
    // Signals
    SD_BUS_SIGNAL("VerifyStatus", "sb", 0),
    SD_BUS_SIGNAL("EnrollStatus", "sibs", 0),
    SD_BUS_SIGNAL("FingerprintStatus", "siss", 0),
    SD_BUS_SIGNAL("FingerprintEnrollStatus", "siss", 0),
    
    // Properties (for now, just return static values)
    SD_BUS_PROPERTY("Version", "s", nullptr, 0, SD_BUS_VTABLE_PROPERTY_CONST),
    SD_BUS_PROPERTY("CameraReady", "b", nullptr, 0, SD_BUS_VTABLE_PROPERTY_CONST),
    SD_BUS_PROPERTY("ModelsLoaded", "i", nullptr, 0, SD_BUS_VTABLE_PROPERTY_CONST),
    
    SD_BUS_VTABLE_END
};

// ============================================================================
// DBusServer Implementation
// ============================================================================

std::unique_ptr<DBusServer> DBusServer::create() {
    sd_bus* bus = nullptr;
    // Use system bus for daemon (accessible from PAM and all users)
    int ret = sd_bus_open_system(&bus);
    
    if (ret < 0) {
        syslog(LOG_ERR, "Failed to open system bus: %s", strerror(-ret));
        return nullptr;
    }

    auto server = std::unique_ptr<DBusServer>(new DBusServer(bus));
    
    // Create auth manager
    server->auth_manager_ = std::make_unique<AuthManager>();
    g_auth_manager = server->auth_manager_.get();
    
     // Create enrollment manager
     server->enroll_manager_ = std::make_unique<EnrollmentManager>();
     
     // Initialize FingerprintManager singleton for fingerprint authentication
     faceid::daemon::FingerprintManager& fp_mgr = faceid::daemon::FingerprintManager::instance();
     bool fp_available = fp_mgr.initialize();
     if (fp_available) {
         syslog(LOG_INFO, "✓ Fingerprint device initialized");
     } else {
         syslog(LOG_WARNING, "⚠ Fingerprint device unavailable (continuing without fingerprint auth)");
     }
     
     // Initialize FaceDetectorManager singleton for face detection and recognition
     FaceDetectorManager& face_mgr = FaceDetectorManager::instance();
     bool face_available = face_mgr.initialize();
     if (face_available) {
         syslog(LOG_INFO, "✓ Face detector models loaded (YuNet + SFace)");
     } else {
         syslog(LOG_ERR, "✗ Face detector models failed to load - face recognition will not work");
         // Don't exit - allow daemon to start for debugging
         // return nullptr;
     }
     
     if (server->register_interfaces() < 0) {
        syslog(LOG_ERR, "Failed to register D-Bus interfaces");
        return nullptr;
    }

    // Claim the bus name
    ret = sd_bus_request_name(bus, "org.freedesktop.FaceID", 0);
    if (ret < 0) {
        syslog(LOG_ERR, "Failed to request bus name: %s", strerror(-ret));
        return nullptr;
    }

    syslog(LOG_INFO, "✓ D-Bus service registered (org.freedesktop.FaceID)");
    return server;
}

DBusServer::~DBusServer() {
    // Stop any ongoing authentication before destroying
    if (auth_manager_) {
        auth_manager_->verify_stop();
    }
    
    if (bus_) {
        sd_bus_release_name(bus_, "org.freedesktop.FaceID");
        sd_bus_unref(bus_);
    }
}

bool DBusServer::wait_event(int timeout_ms) {
    if (!bus_) return false;
    
    return sd_bus_wait(bus_, timeout_ms * 1000) > 0;
}

int DBusServer::register_interfaces() {
    // Register the Device object and its virtual table
    int ret = sd_bus_add_object_vtable(bus_,
        nullptr,                                    // slot (we don't use it)
        "/org/freedesktop/FaceID/Device",          // object path
        "org.freedesktop.FaceID.Device",           // interface name
        device_vtable,                              // virtual table with methods
        this);                                      // userdata (passed to handlers)
    
    if (ret < 0) {
        syslog(LOG_ERR, "Failed to register object vtable: %s", strerror(-ret));
        return ret;
    }
    
    syslog(LOG_INFO, "✓ D-Bus interface registered with method handlers");
    return 0;
}
