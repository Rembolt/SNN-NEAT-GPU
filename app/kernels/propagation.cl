/*
Warp per Neuron 32 kernels for connections
Pick neuron from spiked only list and go through connections CSR to add
weight to dellay list
    arrival slot = (current tick + delay) % delay_max
    target = arrival slot * N_neurons + target_neuron_id
    mark dirty bit updated list
    if input neuron -> add neuron_type_target * weight
    else ->
        add weight to target on delay circular list
        
    if output neuron -> atomic add one count to output buffer slot selected by neuron_kind_target


Needed Buffers:
Neuron param 1d buffers:
-neuron type


Organization buffers:
-current tick
-CSR column index
-CSR row pointer 
-weights list
-delays list
-circular updated list dirty bit(2d, N/32 | max delay)(who recieved charge)
-spiked only list
-circular delay list(2d, column size = max delay)
-output buffer (x bins, y bins, right click, left click)

global Buffers:
-Num of Neurons
-max trace tick
-max delay

*/

//OpenCL lacks float atomics. Helper compare-and-swap on float's bit pattern
inline void atomic_add_float(__global float* target, const float value)
{
    volatile __global uint* target_as_uint = (volatile __global uint*)target;
    uint old_value = *target_as_uint;
    uint expected_value;

    do {
        expected_value = old_value;
        const float next_float = as_float(expected_value) + value;
        old_value = atomic_cmpxchg(target_as_uint, expected_value, as_uint(next_float));
    } while (old_value != expected_value);
}

__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void propagation(
    const uint            current_tick,
    const uint            num_neurons,
    const uint            words_per_row,
    const uint            max_delay,
    const uint            num_input_neurons,
    const uint            num_output_slots,

    __global const uint*  csr_row_ptr,
    __global const uint*  csr_col_idx,
    __global const float* weights,
    __global const float* delays,

    __global const uint*  spiked_only_list,
    __global const uint*  spiked_only_list_offset,

    __global float*       delay_charges,
    __global uint*        dirty_bits,

    __global const float* input_buffer,
    __global uint*        output_buffer,

    __global const uint*  neuron_kind,
    __global const uint*  neuron_kind_target,
    __global uint*        inhibitory_spike_rate,
    __global uint*        excitatory_spike_rate)
{
    const uint spike_index = (uint)get_group_id(0);
    const uint lane_id = (uint)get_local_id(0);
    const uint spiked_count = spiked_only_list_offset[0];

    if (spike_index >= spiked_count || max_delay == 0u) {
        return;
    }

    //Pick neuron from spiked only list and go through connections CSR to add
    //weight to dellay list
    const uint source_neuron_id = spiked_only_list[spike_index];
    if (source_neuron_id >= num_neurons) {
        return;
    }

    const uint source_kind = neuron_kind[source_neuron_id];
    const uint source_target = neuron_kind_target[source_neuron_id];
    //if input, get effect from input buffer
    float source_effect = 1.0f;
    if (source_kind == 1u) {
        source_effect = (source_target < num_input_neurons)
            ? input_buffer[source_target]
            : 0.0f;
    }

    //if output neuron -> first lane adds one count to its target output slot
    if (lane_id == 0u && source_kind == 2u && source_target < num_output_slots) {
        atomic_inc(&output_buffer[source_target]);
    }

    const uint row_begin = csr_row_ptr[source_neuron_id];
    const uint row_end = csr_row_ptr[source_neuron_id + 1u];
    //each lane takes care of 1/32 of its connections
    for (uint edge_index = row_begin + lane_id; edge_index < row_end; edge_index += 32u) {
        const uint target_neuron_id = csr_col_idx[edge_index];
        if (target_neuron_id >= num_neurons) {
            continue;
        }

        //arrival slot = (current tick + delay) % delay_max
        const uint delay_ticks = max((uint)max(delays[edge_index], 0.0f), 1u);
        const uint arrival_slot = (current_tick + delay_ticks) % max_delay;

        //target = arrival slot * N_neurons + target_neuron_id
        const uint delay_charge_index = arrival_slot * num_neurons + target_neuron_id;
        const float charge = source_effect * weights[edge_index];

        if (weights[edge_index] < 0.0f) {
            atomic_inc(&inhibitory_spike_rate[0]);
        } else if (weights[edge_index] > 0.0f) {
            atomic_inc(&excitatory_spike_rate[0]);
        }

        //add weight to target on delay circular list
        atomic_add_float(&delay_charges[delay_charge_index], charge);

        //mark dirty bit updated list
        const uint word_index = target_neuron_id >> 5;
        const uint bit_index = target_neuron_id & 31u;
        const uint dirty_index = arrival_slot * words_per_row + word_index;
        atomic_or(&dirty_bits[dirty_index], 1u << bit_index);
    }
}
