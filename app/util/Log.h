#pragma once

#include <string>
#include <string_view>

namespace Util::Logs {

void write(std::string_view message);

}  // namespace Util::Logs

namespace Util::Bugs {

void write(std::string_view message);

}  // namespace Util::Bugs

namespace Util::Errors {

void write(std::string_view message);
[[noreturn]] void stop(std::string message);

}  // namespace Util::Errors
