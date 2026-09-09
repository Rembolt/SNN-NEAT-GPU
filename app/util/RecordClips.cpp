#include "App.h"
#include "Constants.h"
#include "util/Log.h"
#include "util/Util.h"

#include <filesystem>
#include <string>
#include <vector>

namespace {

std::size_t countMouseInputClipFiles()
{
    namespace fs = std::filesystem;
    const fs::path clips_dir = fs::path(SNN_GPU_PROJECT_ROOT) / mouse_clip::k_dir_name;
    if (!fs::is_directory(clips_dir)) {
        return 0;
    }

    std::size_t count = 0;
    for (const fs::directory_entry& entry : fs::directory_iterator(clips_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() == mouse_clip::k_extension) {
            ++count;
        }
    }
    return count;
}

}  // namespace

void Util::RecordClips::populateMouseInputClipFolder(
    App& app,
    unsigned long tick_count,
    std::chrono::milliseconds tick_target,
    int clicks_required,
    int target_clip_count,
    int record_attempts_per_clip,
    const std::function<void(const std::vector<HostMouseInput>&)>& save_clip)
{
    const std::size_t existing_count = countMouseInputClipFiles();
    if (existing_count >= static_cast<std::size_t>(target_clip_count)) {
        return;
    }

    const int clips_needed =
        target_clip_count - static_cast<int>(existing_count);
    const int max_total_attempts = clips_needed * record_attempts_per_clip * 10;

    Util::Logs::write(
        "RecordClips: recording "
        + std::to_string(clips_needed)
        + " mouse input clip(s) for "
        + std::to_string(tick_count)
        + " ticks (~"
        + std::to_string(
              std::chrono::duration_cast<std::chrono::seconds>(tick_target * tick_count).count())
        + "s at 16ms/tick). Each clip needs >="
        + std::to_string(clicks_required)
        + " clicks. Move the mouse and click.\n");

    int saved_count = 0;
    for (int attempt = 0; saved_count < clips_needed && attempt < max_total_attempts; ++attempt) {
        std::vector<HostMouseInput> input_history;
        for (int record_attempt = 0; record_attempt < record_attempts_per_clip; ++record_attempt) {
            input_history =
                app.recordedInputForTraining(tick_count, tick_target, clicks_required);
            if (!input_history.empty()) {
                break;
            }
        }

        if (input_history.empty()) {
            continue;
        }

        save_clip(input_history);
        ++saved_count;
        Util::Logs::write(
            "RecordClips: saved clip "
            + std::to_string(saved_count)
            + "/"
            + std::to_string(clips_needed)
            + "\n");
    }

    if (saved_count < clips_needed) {
        Util::Errors::stop(
            "RecordClips: recorded "
            + std::to_string(saved_count)
            + "/"
            + std::to_string(clips_needed)
            + " valid clips (each needs >="
            + std::to_string(clicks_required)
            + " clicks) in "
            + (std::filesystem::path(SNN_GPU_PROJECT_ROOT) / mouse_clip::k_dir_name).string());
    }
}
