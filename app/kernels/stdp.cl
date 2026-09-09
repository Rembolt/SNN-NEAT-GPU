/*
Warp per neuron 32 kernels for each connection
Checks spiked only list neurons and navigate CSC connections trace
checks spiked only list
goes to the csc and checks all connections
calculate their trace by using:
    ticks_since_last_fire = current_tick - last_fire_tick_pre + delay_of_connection
    if(ticks_since_last_fire >= trace_lut_size) exit
    (else) trace = trace_lut[ticks_since_last_fire]

error_feedback (packed after HostMouseInput in d_input):
last_click_distance_index distance [0,1] -> accuracy = 1 - distance
feedback_temporal_decay_index temporal_decay [0,1]
reward_factor = click_distance_accuracy * click_distance_weight * temporal_decay
connection raw weight change = (potentiation_bias * trace) - (depression_bias(1-trace))
connection weight change *= reward_factor 

neuron buffers:
-Last fire tick(used to calculate trace)

organization buffers:
-current tick
-CSC column index
-CSC row pointer 
-CSC_to_CSR index mapping 
-weights list
-delays list
-potentiation bias list
-depression bias list
-error feedback buffer

global buffers:
-trace decay look-up table
-trace LUT size
-click_distance_weight

*/

__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void stdp(
    const uint            current_tick,
    const uint            num_neurons,
    const uint            max_trace_len,
    const float           click_distance_weight,
    const uint            feedback_temporal_decay_index,
    const uint            last_click_distance_index,

    __global const uint*  csc_row_ptr,
    __global const uint*  csc_col_idx,
    __global const uint*  csc_to_csr_index_mapping,

    __global float*       weights,
    __global const float* delays,
    __global const float* potentiation_bias,
    __global const float* depression_bias,

    __global const float* input_buffer,
    __global const float* trace_lut,

    __global const uint*  spiked_only_list,
    __global const uint*  spiked_only_list_offset,
    __global const uint*  last_fire_tick,
    const uint            num_synapses)
{
    const uint spike_index = (uint)get_group_id(0);
    const uint lane_id = (uint)get_local_id(0);
    const uint spiked_count = spiked_only_list_offset[0];

    if (spike_index >= spiked_count || max_trace_len == 0u) {
        return;
    }

    const uint target_neuron_id = spiked_only_list[spike_index];
    if (target_neuron_id >= num_neurons) {
        return;
    }

    const float last_click_distance =
        clamp(input_buffer[last_click_distance_index], 0.0f, 1.0f);
    const float click_distance_accuracy = 1.0f - last_click_distance;
    const float temporal_decay =
        clamp(input_buffer[feedback_temporal_decay_index], 0.0f, 1.0f);
    const float reward_factor =
        click_distance_accuracy * click_distance_weight * temporal_decay;
    if (reward_factor == 0.0f) {
        return;
    }

    const uint row_begin = csc_row_ptr[target_neuron_id];
    const uint row_end = csc_row_ptr[target_neuron_id + 1u];

    // each lane takes care of 1/32 of the incoming connections
    for (uint csc_index = row_begin + lane_id; csc_index < row_end; csc_index += 32u) {
        const uint source_neuron_id = csc_col_idx[csc_index];
        if (source_neuron_id >= num_neurons) {
            continue;
        }

        const uint pre_fire_tick = last_fire_tick[source_neuron_id];
        if (pre_fire_tick == 0xFFFFFFFFu) {
            continue;
        }

        const uint csr_index = csc_to_csr_index_mapping[csc_index];
        if (csr_index >= num_synapses) {
            continue;
        }
        const uint delay_ticks = max((uint)max(delays[csr_index], 0.0f), 1u);
        const ulong pre_arrival_tick = (ulong)pre_fire_tick + (ulong)delay_ticks;
        if ((ulong)current_tick < pre_arrival_tick) {
            continue;
        }

        const ulong trace_distance = (ulong)current_tick - pre_arrival_tick;
        if (trace_distance >= (ulong)max_trace_len) {
            continue;
        }

        const float trace = trace_lut[(uint)trace_distance];
        const float raw_weight_change =
            (potentiation_bias[csr_index] * trace)
            - (depression_bias[csr_index] * (1.0f - trace));
        weights[csr_index] += raw_weight_change * reward_factor;
    }
}
