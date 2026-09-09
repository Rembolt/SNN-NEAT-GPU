#pragma once

#include <CL/cl.h>

class CLBoilerplate {
public:
    CLBoilerplate();
    ~CLBoilerplate();

    void printDevice() const;

    cl_device_id     device() const { return device_; }
    cl_context       context() const { return context_; }
    cl_command_queue queue() const { return queue_; }

private:
    cl_platform_id   platform_{};
    cl_device_id     device_{};
    cl_context       context_{};
    cl_command_queue queue_{};
    cl_program       program_{};
    cl_kernel        kernel_{};
};
