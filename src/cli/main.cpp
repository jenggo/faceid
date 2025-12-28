#include <iostream>
#include <string>
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
        return cmd_test(argv[2]);
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
    
    std::cerr << "Unknown command: " << command << std::endl;
    print_usage();
    return 1;
}
