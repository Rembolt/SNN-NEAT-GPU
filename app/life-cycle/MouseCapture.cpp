#include "MouseCapture.h"
#include "Constants.h"
#include "util/Log.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <algorithm>
#include <atomic>
#include <sstream>
#include <thread>

namespace {
HHOOK g_mouse_hook = nullptr;
//atomic variable dont get duplicated by multiple threads, meaning no load -> add -> store
std::atomic<long> g_scroll_up_events{0};
std::atomic<long> g_scroll_down_events{0};
std::atomic<bool> g_hook_thread_running{false};
std::atomic<bool> g_hook_installed{false};
std::thread g_hook_thread{};
DWORD g_hook_thread_id = 0u;

// Low-level mouse hook: Win32 always calls on mouse event
// See Win32 LowLevelMouseProc — LPARAM points at MSLLHOOKSTRUCT for mouse messages
LRESULT CALLBACK MouseHookCallback(
    int hook_notify_code,
    WPARAM mouse_message_id,
    LPARAM pointer_to_mouse_hook_struct)
{
    if (hook_notify_code >= 0 && mouse_message_id == WM_MOUSEWHEEL) {
        const auto* mouse_hook_details =
            reinterpret_cast<const MSLLHOOKSTRUCT*>(pointer_to_mouse_hook_struct);
        const short wheel_delta = GET_WHEEL_DELTA_WPARAM(mouse_hook_details->mouseData);
        if (wheel_delta > 0) {
            //fetch add for atomic operation
            g_scroll_up_events.fetch_add(1, std::memory_order_relaxed);
        } else if (wheel_delta < 0) {
            //fetch add for atomic operation
            g_scroll_down_events.fetch_add(1, std::memory_order_relaxed);
        }
    }
    //call next hook in chain
    return CallNextHookEx(
        g_mouse_hook,
        hook_notify_code,
        mouse_message_id,
        pointer_to_mouse_hook_struct);
}

void runMouseHookThread()
{
    g_hook_thread_id = GetCurrentThreadId();

    // Ensure this thread has a message queue before installing hook/posting quit.
    MSG warmup_msg{};
    PeekMessageW(&warmup_msg, nullptr, 0, 0, PM_NOREMOVE);

    g_mouse_hook = SetWindowsHookExW(WH_MOUSE_LL, MouseHookCallback, nullptr, 0);
    if (g_mouse_hook == nullptr) {
        Util::Bugs::write("Failed to install low-level mouse hook for wheel events\n");
        g_hook_installed.store(false, std::memory_order_release);
        g_hook_thread_running.store(false, std::memory_order_release);
        return;
    }
    g_hook_installed.store(true, std::memory_order_release);

    //runs until WM_QUIT is posted by releaseMouseHook
    MSG msg{};
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    //release hook
    UnhookWindowsHookEx(g_mouse_hook);
    g_mouse_hook = nullptr;
    g_hook_installed.store(false, std::memory_order_release);
    g_hook_thread_running.store(false, std::memory_order_release);
    g_hook_thread_id = 0u;
}

void setMouseHook()
{
    if (g_hook_thread_running.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    //start thread to run mouse hook
    g_hook_thread = std::thread(runMouseHookThread);
}

void releaseMouseHook()
{
    if (!g_hook_thread_running.load(std::memory_order_acquire)) {
        return;
    }
    //post WM_QUIT to thread to stop it(while loop exits when GetMessageW returns 0)
    if (g_hook_thread_id != 0u) {
        PostThreadMessageW(g_hook_thread_id, WM_QUIT, 0, 0);
    }
    //join thread to ensure it has finished processing all messages
    if (g_hook_thread.joinable()) {
        g_hook_thread.join();
    }
}

//read counter and reset to 0
float consumeScrollEvent(std::atomic<long>& counter)
{
    //exchanges reads last counter value and sets counter to 0;
    return counter.exchange(0, std::memory_order_relaxed) > 0 ? 1.0f : 0.0f;
}
}

MouseCapture::MouseCapture()
    : current_x_(0.0)
    , current_y_(0.0)
    , initialized_(false)
{
}

void MouseCapture::initialize()
{
    if (initialized_) {
        return;
    }

    if (!glfwInit()) {
        Util::Errors::stop("Failed to initialize GLFW");
    }

    setMouseHook();
    h_mouse_input = {};
    updateClickState();
    initialized_ = true;
}

bool MouseCapture::initializeMonitors(std::vector<MonitorDisplayInfo>& out_monitors)
{
    int monitor_count = 0;
    GLFWmonitor** glfw_monitors = glfwGetMonitors(&monitor_count);

    if (!glfw_monitors || monitor_count == 0) {
        Util::Bugs::write("No monitors detected\n");
        return false;
    }

    out_monitors.clear();
    out_monitors.reserve(static_cast<std::size_t>(monitor_count));
    glfw_monitor_handles_by_index_.clear();
    glfw_monitor_handles_by_index_.resize(static_cast<std::size_t>(monitor_count), nullptr);

    for (int i = 0; i < monitor_count; ++i) {
        GLFWmonitor* monitor = glfw_monitors[i];
        MonitorDisplayInfo info{};
        glfw_monitor_handles_by_index_[static_cast<std::size_t>(i)] = monitor;
        info.monitor_idx = i;
        info.monitor_name = std::string(glfwGetMonitorName(monitor));
        std::transform(info.monitor_name.begin(), info.monitor_name.end(), info.monitor_name.begin(), ::tolower);
        
        glfwGetMonitorPos(monitor, &info.virtual_pos_x, &info.virtual_pos_y);

        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        if (!mode) {
            std::ostringstream message;
            message << "Failed to get video mode for monitor " << i << "\n";
            Util::Bugs::write(message.str());
            continue;
        }
        // Virtual dimensions (logical, DPI-scaled)
        info.virtual_width = mode->width;
        info.virtual_height = mode->height;

        // DPI scale
        float content_scale_x = 1.0f;
        float content_scale_y = 1.0f;
        glfwGetMonitorContentScale(monitor, &content_scale_x, &content_scale_y);
        info.dpi_scale_x = content_scale_x;
        info.dpi_scale_y = content_scale_y;

        // Physical dimensions (actual pixels)
        info.physical_width =
            static_cast<int>(static_cast<float>(info.virtual_width) * content_scale_x);
        info.physical_height =
            static_cast<int>(static_cast<float>(info.virtual_height) * content_scale_y);

        // Aspect ratio
        info.aspect_ratio = static_cast<float>(info.physical_width) /
                             static_cast<float>(info.physical_height);

        out_monitors.push_back(info);

        if constexpr (debug::k_debug_logs) {
            std::ostringstream message;
            message << "Monitor " << i << ": " << glfwGetMonitorName(monitor) << "\n"
                    << "  Virtual: " << info.virtual_pos_x << "," << info.virtual_pos_y << " "
                    << info.virtual_width << "x" << info.virtual_height << "\n"
                    << "  Physical: " << info.physical_width << "x" << info.physical_height << "\n"
                    << "  DPI Scale: " << info.dpi_scale_x << "x" << info.dpi_scale_y << "\n";
            Util::Logs::write(message.str());
        }
    }
    if (!out_monitors.empty()) {
        current_monitor_info_ = &out_monitors.front();
    }

    return !out_monitors.empty();
}

void MouseCapture::updateClickState()
{

    bool last_left_button_down = h_mouse_input.left_mouse_clicked >= 0.5f;
    bool last_right_button_down = h_mouse_input.right_mouse_clicked >= 0.5f;

    const bool left_button_down = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
    const bool right_button_down = (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0;
    const bool middle_button_down = (GetAsyncKeyState(VK_MBUTTON) & 0x8000) != 0;

    h_mouse_input.left_click_edge = left_button_down && !last_left_button_down ? 1.0f : 0.0f;
    h_mouse_input.right_click_edge = right_button_down && !last_right_button_down ? 1.0f : 0.0f;

    h_mouse_input.left_mouse_clicked = left_button_down ? 1.0f : 0.0f;
    h_mouse_input.right_mouse_clicked = right_button_down ? 1.0f : 0.0f;
    h_mouse_input.scroll_mouse_clicked = middle_button_down ? 1.0f : 0.0f;
    h_mouse_input.scroll_mouse_up = consumeScrollEvent(g_scroll_up_events);
    h_mouse_input.scroll_mouse_down = consumeScrollEvent(g_scroll_down_events);


}

void MouseCapture::updateRawPosition()
{
    // Get absolute cursor position using platform-specific APIs
    //Windows API
    POINT pt{};
    if (GetCursorPos(&pt)) {
        current_x_ = static_cast<double>(pt.x);
        current_y_ = static_cast<double>(pt.y);
    }
}

bool MouseCapture::hasLeftMonitor(
    int                                    current_monitor_idx,
    const std::vector<MonitorDisplayInfo>& monitors) const
{
    if (current_monitor_idx < 0 ||
        current_monitor_idx >= static_cast<int>(monitors.size())) {
        return true; // Invalid monitor = consider as left
    }

    const MonitorDisplayInfo& monitor = monitors[static_cast<std::size_t>(current_monitor_idx)];
     // Check if current position is within monitor's virtual bounds
    const bool inside =
        (current_x_ >= monitor.virtual_pos_x) &&
        (current_x_ < monitor.virtual_pos_x + monitor.virtual_width) &&
        (current_y_ >= monitor.virtual_pos_y) &&
        (current_y_ < monitor.virtual_pos_y + monitor.virtual_height);

    return !inside;
}

int MouseCapture::getCurrentMonitor(
    const std::vector<MonitorDisplayInfo>& monitors,
    int                                    current_monitor_idx)
{
    for (std::size_t i = 0; i < monitors.size(); ++i) {
        const MonitorDisplayInfo& monitor = monitors[i];

        const bool inside =
            (current_x_ >= monitor.virtual_pos_x) &&
            (current_x_ < monitor.virtual_pos_x + monitor.virtual_width) &&
            (current_y_ >= monitor.virtual_pos_y) &&
            (current_y_ < monitor.virtual_pos_y + monitor.virtual_height);

        if (inside) {
            current_monitor_info_ = &monitor;
            return static_cast<int>(i);
        }
    }

    return current_monitor_idx; // Not on any monitor
}

void MouseCapture::getMonitorLocalNormalized(const MonitorDisplayInfo& monitor)
{
    // Convert absolute position to monitor-local coordinates
    const double local_x = current_x_ - static_cast<double>(monitor.virtual_pos_x);
    const double local_y = current_y_ - static_cast<double>(monitor.virtual_pos_y);

    // Normalize to [0,1] range
    h_mouse_input.x_norm =
        static_cast<float>(local_x / static_cast<double>(monitor.virtual_width));
    h_mouse_input.y_norm =
        static_cast<float>(local_y / static_cast<double>(monitor.virtual_height));
   
    h_mouse_input.x_norm = std::clamp(h_mouse_input.x_norm, 0.0f, 1.0f);
    h_mouse_input.y_norm = std::clamp(h_mouse_input.y_norm, 0.0f, 1.0f);
}

MouseCapture::~MouseCapture()
{
    if (initialized_) {
        releaseMouseHook();
        glfwTerminate();
        initialized_ = false;
    }
}
