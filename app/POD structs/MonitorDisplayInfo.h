// MonitorDisplayInfo.h
#pragma once

#include <string>

struct MonitorDisplayInfo {
    int          monitor_idx{};
    std::string  monitor_name{};
    
    // Virtual desktop coordinates (logical, DPI-scaled)
    int   virtual_pos_x{};
    int   virtual_pos_y{};
    int   virtual_width{};
    int   virtual_height{};
    
    // Physical framebuffer dimensions (actual pixels)
    int   physical_width{};
    int   physical_height{};
    
    // DPI scaling
    float dpi_scale_x{};
    float dpi_scale_y{};
    float aspect_ratio{};
};
