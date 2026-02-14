#include "dbus_client_cli.h"
#include <syslog.h>
#include <cstring>
#include <iostream>
#include <turbojpeg.h>
#include <csetjmp>

// Constructor
DBusClientCLI::DBusClientCLI() {}

// Destructor
DBusClientCLI::~DBusClientCLI() {
    disconnect();
}

// Singleton instance
DBusClientCLI& DBusClientCLI::instance() {
    static DBusClientCLI client;
    return client;
}

// Connect to system D-Bus
bool DBusClientCLI::connect() {
    if (bus_ != nullptr) {
        return true;  // Already connected
    }
    
    sd_bus_error error = SD_BUS_ERROR_NULL;
    int r = sd_bus_open_system(&bus_);
    if (r < 0) {
        std::cerr << "Failed to connect to system D-Bus: " << strerror(-r) << std::endl;
        sd_bus_error_free(&error);
        return false;
    }
    
    return true;
}

// Disconnect from system D-Bus
void DBusClientCLI::disconnect() {
    if (bus_ != nullptr) {
        sd_bus_unref(bus_);
        bus_ = nullptr;
    }
}

// Encode frame as JPEG using libturbojpeg (since we can't use OpenCV)
std::vector<uint8_t> DBusClientCLI::encode_frame_as_jpeg(const faceid::Image& frame) {
    std::vector<uint8_t> result;
    
    if (frame.empty()) {
        std::cerr << "Cannot encode empty frame" << std::endl;
        return result;
    }
    
    // Use libturbojpeg for fast compression
    // We use JPEG instead of PNG for performance (faster than libpng)
    
    tjhandle compressor = tjInitCompress();
    if (compressor == nullptr) {
        std::cerr << "Failed to initialize libturbojpeg compressor" << std::endl;
        return result;
    }
    
    unsigned char* jpeg_buf = nullptr;
    unsigned long jpeg_size = 0;
    
    int rows = frame.rows();
    int cols = frame.cols();
    
    // Assuming frame is BGR (3 channels), matching libyuv output
    int subsampling = TJSAMP_444;  // 4:4:4 for quality
    int quality = 90;               // High quality for ML accuracy
    
    int r = tjCompress2(compressor,
        (unsigned char*)frame.data(),
        cols,                        // width
        0,                          // pitch (0 = default)
        rows,                       // height
        TJPF_BGR,                   // pixel format (BGR for OpenCV compatibility)
        &jpeg_buf,
        &jpeg_size,
        subsampling,
        quality,
        TJFLAG_FASTDCT
    );
    
    if (r < 0) {
        std::cerr << "JPEG compression failed: " << tjGetErrorStr2(compressor) << std::endl;
        tjDestroy(compressor);
        return result;
    }
    
    // Copy to result vector
    result.assign(jpeg_buf, jpeg_buf + jpeg_size);
    
    tjFree(jpeg_buf);
    tjDestroy(compressor);
    
    return result;
}

// Detect faces in a frame
DBusClientCLI::DetectResult DBusClientCLI::detect_face(const std::vector<uint8_t>& jpeg_data) {
    DetectResult result = {{}, 0.0};
    
    if (!is_connected()) {
        std::cerr << "Not connected to D-Bus daemon" << std::endl;
        return result;
    }
    
    sd_bus_error error = SD_BUS_ERROR_NULL;
    sd_bus_message* call = nullptr;
    sd_bus_message* reply = nullptr;
    
    // Build method call message manually to handle byte array correctly
    int r = sd_bus_message_new_method_call(bus_,
        &call,
        "org.freedesktop.FaceID",                  // service
        "/org/freedesktop/FaceID/Device",          // object path
        "org.freedesktop.FaceID.Device",           // interface
        "DetectFace"                               // method
    );
    
    if (r < 0) {
        std::cerr << "Failed to create method call: " << strerror(-r) << std::endl;
        return result;
    }
    
    // Append byte array using sd_bus_message_append_array
    r = sd_bus_message_append_array(call, 'y', jpeg_data.data(), jpeg_data.size());
    if (r < 0) {
        std::cerr << "Failed to append JPEG data: " << strerror(-r) << std::endl;
        sd_bus_message_unref(call);
        return result;
    }
    
    // Make the call
    r = sd_bus_call(bus_, call, 0, &error, &reply);
    sd_bus_message_unref(call);
    
    if (r < 0) {
        std::cerr << "DetectFace call failed: " << (error.message ? error.message : strerror(-r)) << std::endl;
        sd_bus_error_free(&error);
        return result;
    }
    
    // Parse response
    // Expected: array of (iiii) bounding boxes, then double quality
    r = sd_bus_message_enter_container(reply, 'a', "(iiii)");
    if (r < 0) {
        std::cerr << "Failed to parse DetectFace response" << std::endl;
        sd_bus_message_unref(reply);
        return result;
    }
    
    // Read each bounding box
    int x, y, width, height;
    while (sd_bus_message_read(reply, "(iiii)", &x, &y, &width, &height) > 0) {
        result.boxes.push_back(faceid::Rect(x, y, width, height));
    }
    
    // Exit array container
    sd_bus_message_exit_container(reply);
    
    // Read quality metric
    sd_bus_message_read(reply, "d", &result.quality);
    
    sd_bus_message_unref(reply);
    return result;
}

// Generate embedding for a face
DBusClientCLI::EmbeddingResult DBusClientCLI::generate_embedding(
    const std::vector<uint8_t>& jpeg_data,
    const faceid::Rect& bbox) {
    
    EmbeddingResult result;
    
    if (!is_connected()) {
        std::cerr << "Not connected to D-Bus daemon" << std::endl;
        return result;
    }
    
    sd_bus_error error = SD_BUS_ERROR_NULL;
    sd_bus_message* call = nullptr;
    sd_bus_message* reply = nullptr;
    
    // Build method call message manually
    int r = sd_bus_message_new_method_call(bus_,
        &call,
        "org.freedesktop.FaceID",
        "/org/freedesktop/FaceID/Device",
        "org.freedesktop.FaceID.Device",
        "GenerateEmbedding"
    );
    
    if (r < 0) {
        std::cerr << "Failed to create method call: " << strerror(-r) << std::endl;
        return result;
    }
    
    // Append byte array
    r = sd_bus_message_append_array(call, 'y', jpeg_data.data(), jpeg_data.size());
    if (r < 0) {
        std::cerr << "Failed to append JPEG data: " << strerror(-r) << std::endl;
        sd_bus_message_unref(call);
        return result;
    }
    
    // Append bounding box struct (iiii)
    r = sd_bus_message_append(call, "(iiii)", bbox.x, bbox.y, bbox.width, bbox.height);
    if (r < 0) {
        std::cerr << "Failed to append bounding box: " << strerror(-r) << std::endl;
        sd_bus_message_unref(call);
        return result;
    }
    
    // Make the call
    r = sd_bus_call(bus_, call, 0, &error, &reply);
    sd_bus_message_unref(call);
    
    if (r < 0) {
        std::cerr << "GenerateEmbedding call failed: " << (error.message ? error.message : strerror(-r)) << std::endl;
        sd_bus_error_free(&error);
        return result;
    }
    
    // Parse response (byte array = embedding)
    const uint8_t* embedding_bytes = nullptr;
    size_t embedding_size = 0;
    
    r = sd_bus_message_read_array(reply, 'y', (const void**)&embedding_bytes, &embedding_size);
    if (r < 0) {
        std::cerr << "Failed to parse GenerateEmbedding response" << std::endl;
        sd_bus_message_unref(reply);
        return result;
    }
    
    // Convert bytes to float array
    const float* float_ptr = (const float*)embedding_bytes;
    size_t float_count = embedding_size / sizeof(float);
    result.embedding.assign(float_ptr, float_ptr + float_count);
    
    sd_bus_message_unref(reply);
    return result;
}

// Verify a face against stored embeddings
DBusClientCLI::VerifyResult DBusClientCLI::verify_face(
    const std::string& username,
    const std::vector<uint8_t>& jpeg_data) {
    
    VerifyResult result = {false, 0.0};
    
    if (!is_connected()) {
        std::cerr << "Not connected to D-Bus daemon" << std::endl;
        return result;
    }
    
    sd_bus_error error = SD_BUS_ERROR_NULL;
    sd_bus_message* call = nullptr;
    sd_bus_message* reply = nullptr;
    
    // Build method call message manually
    int r = sd_bus_message_new_method_call(bus_,
        &call,
        "org.freedesktop.FaceID",
        "/org/freedesktop/FaceID/Device",
        "org.freedesktop.FaceID.Device",
        "VerifyFace"
    );
    
    if (r < 0) {
        std::cerr << "Failed to create method call: " << strerror(-r) << std::endl;
        return result;
    }
    
    // Append username
    r = sd_bus_message_append(call, "s", username.c_str());
    if (r < 0) {
        std::cerr << "Failed to append username: " << strerror(-r) << std::endl;
        sd_bus_message_unref(call);
        return result;
    }
    
    // Append byte array
    r = sd_bus_message_append_array(call, 'y', jpeg_data.data(), jpeg_data.size());
    if (r < 0) {
        std::cerr << "Failed to append JPEG data: " << strerror(-r) << std::endl;
        sd_bus_message_unref(call);
        return result;
    }
    
    // Make the call
    r = sd_bus_call(bus_, call, 0, &error, &reply);
    sd_bus_message_unref(call);
    
    if (r < 0) {
        std::cerr << "VerifyFace call failed: " << (error.message ? error.message : strerror(-r)) << std::endl;
        sd_bus_error_free(&error);
        return result;
    }
    
    // Parse response (bool match, double confidence)
    int match_int = 0;
    sd_bus_message_read(reply, "bd", &match_int, &result.confidence);
    result.match = (match_int != 0);
    
    sd_bus_message_unref(reply);
    return result;
}

// Save embedding for a user
bool DBusClientCLI::save_embedding(
    const std::string& username,
    const std::vector<float>& embedding,
    const std::string& pose) {
    
    if (!is_connected()) {
        std::cerr << "Not connected to D-Bus daemon" << std::endl;
        return false;
    }
    
    sd_bus_error error = SD_BUS_ERROR_NULL;
    
    // Convert embedding to bytes
    std::vector<uint8_t> embedding_bytes;
    embedding_bytes.reserve(embedding.size() * sizeof(float));
    for (float val : embedding) {
        uint8_t* ptr = (uint8_t*)&val;
        embedding_bytes.insert(embedding_bytes.end(), ptr, ptr + sizeof(float));
    }
    
    // Build method call message manually
    sd_bus_message* call = nullptr;
    int r = sd_bus_message_new_method_call(bus_,
        &call,
        "org.freedesktop.FaceID",
        "/org/freedesktop/FaceID/Device",
        "org.freedesktop.FaceID.Device",
        "SaveEmbedding"
    );
    
    if (r < 0) {
        std::cerr << "Failed to create method call: " << strerror(-r) << std::endl;
        return false;
    }
    
    // Append username
    r = sd_bus_message_append(call, "s", username.c_str());
    if (r < 0) {
        std::cerr << "Failed to append username: " << strerror(-r) << std::endl;
        sd_bus_message_unref(call);
        return false;
    }
    
    // Append embedding byte array
    r = sd_bus_message_append_array(call, 'y', embedding_bytes.data(), embedding_bytes.size());
    if (r < 0) {
        std::cerr << "Failed to append embedding: " << strerror(-r) << std::endl;
        sd_bus_message_unref(call);
        return false;
    }
    
    // Append pose
    r = sd_bus_message_append(call, "s", pose.c_str());
    if (r < 0) {
        std::cerr << "Failed to append pose: " << strerror(-r) << std::endl;
        sd_bus_message_unref(call);
        return false;
    }
    
    // Make the call
    r = sd_bus_call(bus_, call, 0, &error, nullptr);
    sd_bus_message_unref(call);
    
    if (r < 0) {
        std::cerr << "SaveEmbedding call failed: " << (error.message ? error.message : strerror(-r)) << std::endl;
        sd_bus_error_free(&error);
        return false;
    }
    
    return true;
}
