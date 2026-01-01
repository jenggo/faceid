#include <iostream>
#include "commands.h"
#include "config_paths.h"

namespace faceid {

void print_usage() {
    std::cout << "FaceID - Linux Face Authentication System" << std::endl;
    std::cout << "Version: " << VERSION << std::endl << std::endl;
    std::cout << "Usage: faceid <command> [options]" << std::endl << std::endl;
    std::cout << "Commands:" << std::endl;
    std::cout << "  add <username> [face_id]                     Add face model for user (default: 'default')" << std::endl;
    std::cout << "  remove <username> [face_id]                  Remove specific face or all faces" << std::endl;
    std::cout << "  list [username]                              List all enrolled users or user's faces" << std::endl;
    std::cout << "  test <username> [--auto-adjust]              Test face recognition with live camera" << std::endl;
    std::cout << "  image test --enroll <img> --test <img>       Test detection/recognition on static images" << std::endl;
    std::cout << "  show                                         Show live camera view with face detection" << std::endl;
    std::cout << "  bench <directory>                            Benchmark recognition models in directory" << std::endl;
    std::cout << "  use <model_path>                             Switch detection or recognition model" << std::endl;
    std::cout << "  devices                                      List available camera devices" << std::endl;
    std::cout << "  version                                      Show version information" << std::endl;
    std::cout << "  help                                         Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  faceid add jenggo                                   # Add default face" << std::endl;
    std::cout << "  faceid add jenggo glasses                           # Add face with glasses" << std::endl;
    std::cout << "  faceid test jenggo                                  # Test recognition for jenggo" << std::endl;
    std::cout << "  faceid test jenggo --auto-adjust                    # Auto-tune confidence & threshold" << std::endl;
    std::cout << "  faceid list jenggo                                  # List all faces for jenggo" << std::endl;
    std::cout << "  faceid remove jenggo glasses                        # Remove 'glasses' face only" << std::endl;
    std::cout << "  faceid remove jenggo                                # Remove ALL faces for jenggo" << std::endl;
    std::cout << "  faceid show                                         # Live camera preview with detection" << std::endl;
    std::cout << "  faceid image test --enroll a.jpg --test b.jpg       # Test recognition on static images" << std::endl;
    std::cout << "  faceid bench /tmp/models                            # Benchmark models (uses embedded image)" << std::endl;
    std::cout << "  faceid bench --image face.jpg /tmp/models           # Benchmark with custom test image" << std::endl;
    std::cout << "  faceid use $(pwd)/models/mnet-retinaface.param      # Switch to RetinaFace detection model" << std::endl;
    std::cout << "  faceid use /path/to/sface_2021dec_int8bq.ncnn.param # Switch to SFace recognition model" << std::endl;
}

} // namespace faceid
