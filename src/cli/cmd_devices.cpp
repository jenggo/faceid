#include <iostream>
#include "../camera.h"

namespace faceid {

using namespace faceid;

int cmd_devices() {
    auto devices = Camera::listDevices();
    if (devices.empty()) {
        std::cerr << "No camera devices found" << std::endl;
        return 1;
    }
    
    std::cout << "Available camera devices:" << std::endl;
    for (const auto& device : devices) {
        std::cout << "  " << device << std::endl;
    }
    return 0;
}

} // namespace faceid
