#include "util/Log.h"

#include <iostream>
#include <mutex>
#include <stdexcept>

namespace {

std::mutex& logMutex()
{
    static std::mutex mutex;
    return mutex;
}

void writeLocked(std::ostream& out, std::string_view message)
{
    std::lock_guard<std::mutex> lock(logMutex());
    out << message;
    if (message.empty() || message.back() != '\n') {
        out << '\n';
    }
    out.flush();
}

}  // namespace

namespace Util::Logs {

void write(std::string_view message)
{
    writeLocked(std::cout, message);
}

}  // namespace Util::Logs

namespace Util::Bugs {

void write(std::string_view message)
{
    writeLocked(std::cerr, message);
}

}  // namespace Util::Bugs

namespace Util::Errors {

void write(std::string_view message)
{
    writeLocked(std::cerr, message);
}

[[noreturn]] void stop(std::string message)
{
    throw std::runtime_error(std::move(message));
}

}  // namespace Util::Errors
