#include "MouseOutput.h"
#include "Constants.h"
#include "MonitorDisplayInfo.h"
#include "util/Log.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <sstream>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

static float parabolicSubBinOffset(int winner, const std::vector<float>& bins, int length)
{
    if (length <= 1) {
        return 0.0f;
    }

    winner = std::clamp(winner, 0, length - 1);
    if (winner == 0 || winner == length - 1) {
        return 0.0f;
    }

    const float y_minus = bins[static_cast<std::size_t>(winner - 1)];
    const float y_zero = bins[static_cast<std::size_t>(winner)];
    const float y_plus = bins[static_cast<std::size_t>(winner + 1)];
    const float denom = y_minus - 2.0f * y_zero + y_plus;
    if (std::fabs(denom) < 1e-6f) {
        return 0.0f;
    }

    return 0.5f * (y_minus - y_plus) / denom;
}

static float binNormalized(int winner, float delta, int length)
{
    if (length <= 0) {
        return 0.5f;
    }
    return std::clamp(
        (static_cast<float>(winner) + delta + 0.5f) / static_cast<float>(length),
        0.0f,
        1.0f);
}


#ifdef _WIN32
// Minimal message handler for the click-through overlay window
static LRESULT CALLBACK GhostOverlayWndProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam)
{
    return DefWindowProcW(hwnd, msg, wparam, lparam);
}

//blends the pixel with the color
static void blendPixelPremultiplied(
    std::uint32_t& dst,
    int r,
    int g,
    int b,
    int a)
{
    // UpdateLayeredWindow expects premultiplied alpha (ARGB in memory)
    const std::uint32_t srcA = static_cast<std::uint32_t>(std::clamp(a, 0, 255));
    const std::uint32_t srcR = static_cast<std::uint32_t>(std::clamp((r * static_cast<int>(srcA)) / 255, 0, 255));
    const std::uint32_t srcG = static_cast<std::uint32_t>(std::clamp((g * static_cast<int>(srcA)) / 255, 0, 255));
    const std::uint32_t srcB = static_cast<std::uint32_t>(std::clamp((b * static_cast<int>(srcA)) / 255, 0, 255));

    const std::uint32_t dstB = dst & 0xFFu;
    const std::uint32_t dstG = (dst >> 8) & 0xFFu;
    const std::uint32_t dstR = (dst >> 16) & 0xFFu;
    const std::uint32_t dstA = (dst >> 24) & 0xFFu;

    const std::uint32_t invA = 255u - srcA;
    const std::uint32_t outA = srcA + (dstA * invA) / 255u;
    const std::uint32_t outR = srcR + (dstR * invA) / 255u;
    const std::uint32_t outG = srcG + (dstG * invA) / 255u;
    const std::uint32_t outB = srcB + (dstB * invA) / 255u;

    dst = (outA << 24) | (outR << 16) | (outG << 8) | outB;
}

static void drawGhostSprite(std::vector<std::uint32_t>& pixels, int size_px)
{
    // Build a small ARGB sprite once, then blit(bit-block transfer) it every frame
    std::fill(pixels.begin(), pixels.end(), 0u);

    // center of the sprite
    const float cx = static_cast<float>(size_px) * 0.5f;
    const float cy = static_cast<float>(size_px) * 0.5f;
    const float fill_radius = mouse_output_ui::k_ghost_cursor_radius;

    for (int y = 0; y < size_px; ++y) {
        for (int x = 0; x < size_px; ++x) {
            const float dx = (static_cast<float>(x) + 0.5f) - cx;
            const float dy = (static_cast<float>(y) + 0.5f) - cy;
            const float d = std::sqrt(dx * dx + dy * dy);
            std::uint32_t& pixel = pixels[static_cast<std::size_t>(y) * static_cast<std::size_t>(size_px)
                                         + static_cast<std::size_t>(x)];
            
            // soft inner fill plus a brighter ring so the overlay is easy to spot
            if (d <= fill_radius) {
                blendPixelPremultiplied(pixel, 90, 160, 255, 110);
            } else if (d <= fill_radius + 2.0f) {
                blendPixelPremultiplied(pixel, 200, 230, 255, 200);
            }
        }
    }
}

//blits (bit-block transfer) the sprite to the window
static void blitOverlaySprite(
    HWND hwnd,
    HDC mem_dc,
    int overlay_size,
    float global_x,
    float global_y)
{
    POINT dst_pt{
        static_cast<LONG>(std::lround(global_x) - overlay_size / 2),
        static_cast<LONG>(std::lround(global_y) - overlay_size / 2)
    };
    SIZE wnd_size{overlay_size, overlay_size};
    POINT src_pt{0, 0};
    BLENDFUNCTION blend{};
    blend.BlendOp = AC_SRC_OVER;
    blend.SourceConstantAlpha = 255;
    blend.AlphaFormat = AC_SRC_ALPHA;

    HDC screen_dc = GetDC(nullptr);
    const BOOL ok = UpdateLayeredWindow(
        hwnd, screen_dc, &dst_pt, &wnd_size, mem_dc, &src_pt, 0, &blend, ULW_ALPHA);
    ReleaseDC(nullptr, screen_dc);

    if (!ok) {
        const DWORD err = GetLastError();
        std::ostringstream message;
        message << "MouseOutput: UpdateLayeredWindow failed, err=" << err
                << " pos=(" << dst_pt.x << "," << dst_pt.y << ")\n";
        Util::Bugs::write(message.str());
        return;
    }

    SetWindowPos(
        hwnd,
        HWND_TOPMOST,
        0,
        0,
        0,
        0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW);
}

static void pumpOverlayMessages()
{
    MSG msg{};
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
}
#endif

MouseOutput::MouseOutput() = default;

MouseOutput::~MouseOutput()
{
    destroyGhostOverlay();
}

void MouseOutput::initialize(int x_bins, int y_bins, float axis_decay)
{
    host_ = {};
    host_.x_length = std::max(1, x_bins);
    host_.y_length = std::max(1, y_bins);
    host_.output_axis_decay = std::clamp(axis_decay, 0.0f, 1.0f);
    host_.x.assign(static_cast<std::size_t>(host_.x_length), 0.0f);
    host_.y.assign(static_cast<std::size_t>(host_.y_length), 0.0f);
    normalized_scratch_.clear();
}

void MouseOutput::updateOutput(
    const std::vector<std::uint32_t>& h_output,
    const std::vector<std::uint32_t>& output_slot_neuron_counts)
{
    if (host_.x_length <= 0
        || host_.y_length <= 0
        || host_.x.size() != static_cast<std::size_t>(host_.x_length)
        || host_.y.size() != static_cast<std::size_t>(host_.y_length)) {
        Util::Errors::stop("MouseOutput: Invalid host buffers");
        return;
    }

    const std::size_t position_slots =
        static_cast<std::size_t>(host_.x_length + host_.y_length);
    if (h_output.size() < position_slots) {
        Util::Errors::stop(
            "MouseOutput::updateOutput: h_output.size() ("
            + std::to_string(h_output.size())
            + ") < position slots ("
            + std::to_string(position_slots)
            + ')');
        return;
    }

    if (normalized_scratch_.size() != h_output.size()) {
        normalized_scratch_.assign(h_output.size(), 0.0f);
    } else {
        std::fill(normalized_scratch_.begin(), normalized_scratch_.end(), 0.0f);
    }

    for (std::size_t slot = 0; slot < h_output.size(); ++slot) {
        const std::uint32_t count = slot < output_slot_neuron_counts.size()
            ? output_slot_neuron_counts[slot]
            : 0u;
        if (count == 0u) {
            continue;
        }
        normalized_scratch_[slot] = std::clamp(
            static_cast<float>(h_output[slot]) / static_cast<float>(count),
            0.0f,
            1.0f);
    }

    const std::vector<float>& normalized_output = normalized_scratch_;

    int index = 0;
    int x_winner = 0;
    int y_winner = 0;
    float total_votes = 0.0f;

    for (int i = 0; i < host_.x_length; ++i) {
        const float votes = normalized_output[static_cast<std::size_t>(index++)];

        host_.x[static_cast<std::size_t>(i)] *= host_.output_axis_decay;
        host_.x[static_cast<std::size_t>(i)] += votes;
        total_votes += host_.x[static_cast<std::size_t>(i)];
        if (host_.x[static_cast<std::size_t>(i)] > host_.x[static_cast<std::size_t>(x_winner)]) {
            x_winner = i;
        }
    }

    for (int i = 0; i < host_.y_length; ++i) {
        const float votes = normalized_output[static_cast<std::size_t>(index++)];

        host_.y[static_cast<std::size_t>(i)] *= host_.output_axis_decay;
        host_.y[static_cast<std::size_t>(i)] += votes;
        total_votes += host_.y[static_cast<std::size_t>(i)];
        if (host_.y[static_cast<std::size_t>(i)] > host_.y[static_cast<std::size_t>(y_winner)]) {
            y_winner = i;
        }
    }

    const float winner_mass =
        host_.x[static_cast<std::size_t>(x_winner)]
        + host_.y[static_cast<std::size_t>(y_winner)];
    host_.certainty = total_votes > 0.0f
        ? std::clamp(winner_mass / total_votes, 0.0f, 1.0f)
        : 0.0f;

    const float delta_x = parabolicSubBinOffset(x_winner, host_.x, host_.x_length);
    const float delta_y = parabolicSubBinOffset(y_winner, host_.y, host_.y_length);
    host_.predicted_x = binNormalized(x_winner, delta_x, host_.x_length);
    host_.predicted_y = binNormalized(y_winner, delta_y, host_.y_length);

    host_.right_click = 0.0f;
    host_.left_click = 0.0f;
    if (h_output.size() >= position_slots + 2u) {
        const float rc = normalized_output[position_slots];
        const float lc = normalized_output[position_slots + 1u];
        if (rc != lc) {
            host_.right_click = rc;
            host_.left_click = lc;
        }
    }
}

void MouseOutput::initializeGhostOverlay(const std::vector<MonitorDisplayInfo>& monitors)
{
    destroyGhostOverlay();
    if (monitors.empty()) {
        return;
    }

#ifdef _WIN32
    // 1) Build the ghost sprite pixels.
    overlay_size_ = mouse_output_ui::k_overlay_size_px;
    overlay_pixels_.resize(static_cast<std::size_t>(overlay_size_) * static_cast<std::size_t>(overlay_size_));
    drawGhostSprite(overlay_pixels_, overlay_size_);

    // 2) Create a layered, click-through, top-most popup window.
    HINSTANCE instance = GetModuleHandleW(nullptr);
    const wchar_t* class_name = L"SNNGhostOverlayWindowClass";

    WNDCLASSW wc{};
    wc.lpfnWndProc = GhostOverlayWndProc;
    wc.hInstance = instance;
    wc.lpszClassName = class_name;
    wc.style = CS_HREDRAW | CS_VREDRAW;
    if (RegisterClassW(&wc) == 0 && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
        std::ostringstream message;
        message << "MouseOutput: RegisterClassW failed, err=" << GetLastError() << '\n';
        Util::Bugs::write(message.str());
        return;
    }

    HWND hwnd = CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_TOPMOST | WS_EX_NOACTIVATE,
        class_name,
        L"SNN ghost cursor",
        WS_POPUP,
        0,
        0,
        overlay_size_,
        overlay_size_,
        nullptr,
        nullptr,
        instance,
        nullptr);

    if (!hwnd) {
        std::ostringstream message;
        message << "MouseOutput: CreateWindowExW failed, err=" << GetLastError() << '\n';
        Util::Bugs::write(message.str());
        return;
    }

    // 3) Create a 32-bit DIB section used as the per-frame source buffer.
    HDC screen_dc = GetDC(nullptr);
    HDC mem_dc = CreateCompatibleDC(screen_dc);

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = overlay_size_;
    bmi.bmiHeader.biHeight = -overlay_size_;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    void* dib_bits = nullptr;
    HBITMAP dib = CreateDIBSection(mem_dc, &bmi, DIB_RGB_COLORS, &dib_bits, nullptr, 0);
    HGDIOBJ old_bitmap = SelectObject(mem_dc, dib);

    ReleaseDC(nullptr, screen_dc);

    if (!dib || !dib_bits) {
        if (old_bitmap) {
            SelectObject(mem_dc, old_bitmap);
        }
        if (dib) {
            DeleteObject(dib);
        }
        if (mem_dc) {
            DeleteDC(mem_dc);
        }
        DestroyWindow(hwnd);
        std::ostringstream message;
        message << "MouseOutput: CreateDIBSection failed, err=" << GetLastError() << '\n';
        Util::Bugs::write(message.str());
        return;
    }

    // Sprite is static; upload it once and only move the window each frame
    std::copy(overlay_pixels_.begin(), overlay_pixels_.end(), static_cast<std::uint32_t*>(dib_bits));

    overlay_hwnd_ = hwnd;
    overlay_mem_dc_ = mem_dc;
    overlay_dib_ = dib;
    overlay_old_bitmap_ = old_bitmap;
    overlay_initialized_ = true;

    // Layered popups stay invisible until the first UpdateLayeredWindow call
    const MonitorDisplayInfo& primary = monitors.front();
    const float initial_x = static_cast<float>(primary.virtual_pos_x)
        + static_cast<float>(primary.virtual_width) * 0.5f;
    const float initial_y = static_cast<float>(primary.virtual_pos_y)
        + static_cast<float>(primary.virtual_height) * 0.5f;
    blitOverlaySprite(hwnd, mem_dc, overlay_size_, initial_x, initial_y);
    pumpOverlayMessages();
#else
    (void)monitors;
#endif
}

void MouseOutput::renderGhostOverlayFrame(
    const MonitorDisplayInfo& active_monitor,
    double cursor_global_x,
    double cursor_global_y)
{
    if (!overlay_initialized_) {
        return;
    }

#ifdef _WIN32
    HWND hwnd = static_cast<HWND>(overlay_hwnd_);
    HDC mem_dc = static_cast<HDC>(overlay_mem_dc_);
    if (!hwnd || !mem_dc) {
        return;
    }

    // Keep overlay placement independent from monitor index selection.
    // This avoids freezing when monitor mapping/capture restarts flap.
    const float global_x = static_cast<float>(cursor_global_x);
    const float global_y = static_cast<float>(cursor_global_y);

    blitOverlaySprite(hwnd, mem_dc, overlay_size_, global_x, global_y);
    pumpOverlayMessages();
#else
    (void)active_monitor;
    (void)cursor_global_x;
    (void)cursor_global_y;
#endif
}

void MouseOutput::destroyGhostOverlay()
{
    if (!overlay_initialized_) {
        return;
    }

#ifdef _WIN32
    // Release GDI objects in reverse order of creation.
    if (overlay_mem_dc_ && overlay_old_bitmap_) {
        SelectObject(static_cast<HDC>(overlay_mem_dc_), static_cast<HGDIOBJ>(overlay_old_bitmap_));
    }
    if (overlay_dib_) {
        DeleteObject(static_cast<HGDIOBJ>(overlay_dib_));
    }
    if (overlay_mem_dc_) {
        DeleteDC(static_cast<HDC>(overlay_mem_dc_));
    }
    if (overlay_hwnd_) {
        DestroyWindow(static_cast<HWND>(overlay_hwnd_));
    }
#endif

    overlay_hwnd_ = nullptr;
    overlay_mem_dc_ = nullptr;
    overlay_dib_ = nullptr;
    overlay_old_bitmap_ = nullptr;
    overlay_pixels_.clear();
    overlay_initialized_ = false;
}
