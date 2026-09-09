#include "CLBoilerplate.h"
#include "util/Log.h"

#include <cstdint>
#include <sstream>
#include <vector>

namespace {
std::int64_t scoreDevice(cl_device_id device) {
    cl_device_type type = 0;
    cl_uint computeUnits = 0;
    cl_ulong globalMemBytes = 0;
    cl_uint maxClockMHz = 0;
    cl_bool available = CL_FALSE;

    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemBytes), &globalMemBytes, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(maxClockMHz), &maxClockMHz, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_AVAILABLE, sizeof(available), &available, nullptr);

    if (available != CL_TRUE) {
        return -1;
    }

    std::int64_t score = 0;
    if (type & CL_DEVICE_TYPE_GPU) {
        score += 100000;
    } else if (type & CL_DEVICE_TYPE_ACCELERATOR) {
        score += 60000;
    } else if (type & CL_DEVICE_TYPE_CPU) {
        score += 20000;
    }

    score += static_cast<std::int64_t>(computeUnits) * 1000;
    score += static_cast<std::int64_t>(maxClockMHz) * 10;
    score += static_cast<std::int64_t>(globalMemBytes / (1024ull * 1024ull * 1024ull)) * 100;

    return score;
}
}

CLBoilerplate::CLBoilerplate() {
    cl_int err = clGetPlatformIDs(1, &platform_, nullptr);
    if (err != CL_SUCCESS || !platform_) {
        Util::Errors::stop("clGetPlatformIDs failed");
    }

    cl_uint deviceCount = 0;
    err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);
    if (err != CL_SUCCESS || deviceCount == 0) {
        Util::Errors::stop("No OpenCL device (GPU or CPU) found");
    }

    std::vector<cl_device_id> devices(deviceCount);
    err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr);
    if (err != CL_SUCCESS) {
        Util::Errors::stop("Failed to enumerate OpenCL devices");
    }

    std::int64_t bestScore = -1;
    for (cl_device_id candidate : devices) {
        const std::int64_t candidateScore = scoreDevice(candidate);
        if (candidateScore > bestScore) {
            bestScore = candidateScore;
            device_ = candidate;
        }
    }

    if (!device_) {
        Util::Errors::stop("No usable OpenCL device found");
    }

    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || !context_) {
        Util::Errors::stop("clCreateContext failed");
    }

    queue_ = clCreateCommandQueueWithProperties(context_, device_, nullptr, &err);
    if (err != CL_SUCCESS || !queue_) {
        clReleaseContext(context_);
        context_ = nullptr;
        Util::Errors::stop("clCreateCommandQueueWithProperties failed");
    }
}

void CLBoilerplate::printDevice() const {
    if (!device_) {
        Util::Logs::write("No OpenCL device.\n");
        return;
    }

    char buf[1024];
    size_t len = 0;
    std::ostringstream message;

    if (clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(buf), buf, &len) == CL_SUCCESS) {
        message << "Device name: " << buf << '\n';
    }

    if (clGetDeviceInfo(device_, CL_DEVICE_VENDOR, sizeof(buf), buf, &len) == CL_SUCCESS) {
        message << "Vendor:      " << buf << '\n';
    }

    if (clGetDeviceInfo(device_, CL_DEVICE_VERSION, sizeof(buf), buf, &len) == CL_SUCCESS) {
        message << "OpenCL:      " << buf << '\n';
    }
    Util::Logs::write(message.str());
}

CLBoilerplate::~CLBoilerplate() {
    if (kernel_) {
        clReleaseKernel(kernel_);
    }
    if (program_) {
        clReleaseProgram(program_);
    }
    if (queue_) {
        clReleaseCommandQueue(queue_);
    }
    if (context_) {
        clReleaseContext(context_);
    }
}
