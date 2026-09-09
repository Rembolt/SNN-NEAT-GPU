__kernel void input_to_spiked(
    const uint            current_tick,
    __global const float* d_input,
    __global uchar*       d_spiked,
    __global uint*        d_last_spike,
    __global const uint*  d_neuron_kind,
    __global const uint*  d_neuron_kind_target,
    const uint            num_neurons,
    const uint            num_input_neurons)
{
    const uint neuron_idx = (uint)get_global_id(0);
    if (neuron_idx >= num_neurons) {
        return;
    }

    const uint kind = d_neuron_kind[neuron_idx];
    if (kind == 3u) {
        d_spiked[neuron_idx] = (uchar)1;
        d_last_spike[neuron_idx] = current_tick;
        return;
    }

    if (kind == 1u) {
        const uint input_idx = d_neuron_kind_target[neuron_idx];
        const uchar fired = (input_idx < num_input_neurons && d_input[input_idx] > 0.0f)
            ? (uchar)1
            : (uchar)0;
        d_spiked[neuron_idx] = fired;
        if (fired != (uchar)0) {
            d_last_spike[neuron_idx] = current_tick;
        }
    }
}
