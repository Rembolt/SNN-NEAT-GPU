#pragma once

/*
Captures raw mouse position/button state and normalizes coordinates to monitor space.
Higher-order movement metrics (delta/acceleration/jerk) are derived in ErrorFeedback.
*/

#include <GLFW/glfw3.h>
#include "HostAppParams.h"
#include "MonitorDisplayInfo.h"
#include "HostMouseInput.h"
#include <cmath>
#include <vector>

class MouseCapture {
public:
    

    MouseCapture();
    ~MouseCapture();

    void initialize();
    bool initializeMonitors(std::vector<MonitorDisplayInfo>& out_monitors);
    void updateRawPosition();
    void updateClickState();
    bool hasLeftMonitor(
        int                                  current_monitor_idx,
        const std::vector<MonitorDisplayInfo>& monitors) const;
    int getCurrentMonitor(
        const std::vector<MonitorDisplayInfo>& monitors,
        int                                  current_monitor_idx);
    void getMonitorLocalNormalized(const MonitorDisplayInfo& monitor);

    HostMouseInput h_mouse_input{};

    double currentX() const noexcept { return current_x_; }
    double currentY() const noexcept { return current_y_; }

private:
    std::vector<GLFWmonitor*> glfw_monitor_handles_by_index_{};
    const MonitorDisplayInfo* current_monitor_info_{};
    double      current_x_{};
    double      current_y_{};
    bool        initialized_{};
};
