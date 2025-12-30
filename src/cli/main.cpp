#include <iostream>
#include <string>
#include <vector>
#include "commands.h"
#include "config_paths.h"

using namespace faceid;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    std::string command = argv[1];
    
    if (command == "help" || command == "--help" || command == "-h") {
        print_usage();
        return 0;
    }
    
    if (command == "version" || command == "--version" || command == "-v") {
        std::cout << "FaceID version " << VERSION << std::endl;
        return 0;
    }
    
    if (command == "devices") {
        return cmd_devices();
    }
    
    if (command == "show") {
        return cmd_show();
    }
    
    if (command == "list") {
        if (argc >= 3) {
            return cmd_list(argv[2]);  // List faces for specific user
        }
        return cmd_list();  // List all users
    }
    
    if (command == "add") {
        if (argc < 3) {
            std::cerr << "Error: username required" << std::endl;
            return 1;
        }
        if (argc >= 4) {
            return cmd_add(argv[2], argv[3]);  // username + face_id
        }
        return cmd_add(argv[2]);  // username only (default face_id)
    }
    
    if (command == "remove") {
        if (argc < 3) {
            std::cerr << "Error: username required" << std::endl;
            return 1;
        }
        if (argc >= 4) {
            return cmd_remove(argv[2], argv[3]);  // Remove specific face
        }
        return cmd_remove(argv[2]);  // Remove all faces
    }
    
    if (command == "test") {
        if (argc < 3) {
            std::cerr << "Error: username required" << std::endl;
            return 1;
        }
        
        // Check for --auto-adjust flag
        bool auto_adjust = false;
        if (argc >= 4 && std::string(argv[3]) == "--auto-adjust") {
            auto_adjust = true;
        }
        
        return cmd_test(argv[2], auto_adjust);
    }
    
    if (command == "image") {
        if (argc < 3) {
            std::cerr << "Error: image subcommand required" << std::endl;
            std::cerr << "Usage: faceid image <subcommand> [options]" << std::endl;
            std::cerr << "Subcommands:" << std::endl;
            std::cerr << "  test --enroll <img> --test <img>  Test detection/recognition on images" << std::endl;
            return 1;
        }
        
        std::string subcmd = argv[2];
        
        if (subcmd == "test") {
            std::vector<std::string> args;
            for (int i = 3; i < argc; i++) {
                args.push_back(argv[i]);
            }
            return cmd_test_image(args);
        } else {
            std::cerr << "Unknown image subcommand: " << subcmd << std::endl;
            std::cerr << std::endl;
            std::cerr << "Usage: faceid image <subcommand> [options]" << std::endl;
            std::cerr << "Available subcommands:" << std::endl;
            std::cerr << "  test --enroll <img> --test <img>  Test detection/recognition on images" << std::endl;
            std::cerr << std::endl;
            std::cerr << "Example: faceid image test --enroll single-face.jpg --test two-faces.jpg" << std::endl;
            return 1;
        }
    }
    
    if (command == "bench" || command == "benchmark") {
        if (argc < 3) {
            std::cerr << "Error: model directory required" << std::endl;
            std::cerr << "Usage: faceid bench [--detail] <model_directory>" << std::endl;
            std::cerr << "Example: faceid bench /tmp/models" << std::endl;
            std::cerr << "         faceid bench --detail /tmp/models" << std::endl;
            return 1;
        }
        
        // Check for --detail flag
        bool show_detail = false;
        std::string test_dir;
        
        if (argc >= 4 && std::string(argv[2]) == "--detail") {
            show_detail = true;
            test_dir = argv[3];
        } else if (argc >= 4 && std::string(argv[3]) == "--detail") {
            show_detail = true;
            test_dir = argv[2];
        } else {
            test_dir = argv[2];
        }
        
        return cmd_bench(test_dir, show_detail);
    }
    
    if (command == "use") {
        if (argc < 3) {
            std::cerr << "Error: absolute model path required" << std::endl;
            std::cerr << "Usage: faceid use <absolute_path_to_model>" << std::endl;
            std::cerr << "Example: faceid use /home/user/models/mnet-retinaface.param" << std::endl;
            std::cerr << "         faceid use $(pwd)/models/sface_2021dec_int8bq.ncnn.param" << std::endl;
            return 1;
        }
        return cmd_use(argv[2]);
    }
    
    std::cerr << "Unknown command: " << command << std::endl;
    print_usage();
    return 1;
}
