#pragma once

/*
 * GPU output slots -> normalized HostMouseOutput (see HostMouseOutput.h).
 */

#include <vector>
#include <cstdint>

#include "HostMouseOutput.h"
struct MonitorDisplayInfo;

class MouseOutput {
public:
    MouseOutput();
    ~MouseOutput();

    MouseOutput(const MouseOutput&) = delete;
    MouseOutput& operator=(const MouseOutput&) = delete;
    MouseOutput(MouseOutput&&) = delete;
    MouseOutput& operator=(MouseOutput&&) = delete;

    void initialize(
        int   x_bins,
        int   y_bins,
        float axis_decay);
    void updateOutput(
        const std::vector<std::uint32_t>& h_output,
        const std::vector<std::uint32_t>& output_slot_neuron_counts);

    // Creates an on top transparent overlay window sized to the virtual desktop 
    // so it can draw on whichever monitor the active normalized coords map to
    void initializeGhostOverlay(const std::vector<MonitorDisplayInfo>& monitors);
    // Renders the ghost cursor for the active monitor
    void renderGhostOverlayFrame(
        const MonitorDisplayInfo& active_monitor,
        double cursor_global_x,
        double cursor_global_y);

    [[nodiscard]] const HostMouseOutput& hostOutput() const noexcept { return host_; }

private:
    HostMouseOutput   host_{};
    std::vector<float> normalized_scratch_{};

    bool overlay_initialized_{false};
    void* overlay_hwnd_{nullptr};
    void* overlay_mem_dc_{nullptr};
    void* overlay_dib_{nullptr};
    void* overlay_old_bitmap_{nullptr};
    int   overlay_size_{64};
    std::vector<std::uint32_t> overlay_pixels_{};

    void destroyGhostOverlay();
};
