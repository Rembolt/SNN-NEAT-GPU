/*
Warp per neuron 32 kernels for each connection 
for every neuron check if firing_rate/desired_firing_rate is more drastic than Homeostasis_rate_max_distance
firing_rate = decayed_trace / fire_rate_tau  (trace decays exp(-1/tau) per tick; decay since last delivery run applied here)
(if so) mod = firing_rate/desired_firing_rate
    threshold moves up when overactive and down when underactive
    navigate through CSC connections
        weight moves down when overactive and up when underactive
        potentiation_bias moves down when overactive and up when underactive
        depression_bias moves up when overactive and down when underactive
        
Needed Buffers:
Neuron param 1d buffers:
-Fire rate 
-fira rate target
-threshold (V_Threshold)

Organization buffers:
-ticks between homeostasis
-current tick
-CSC row pointer 
-CSC_to_CSR index mapping 
-weights list
-potentiation bias list
-depression bias list

global Buffers:
-Homeostasis weight mod.
-Homeostasis pot. bias mod.
-Homeostasis dep. bias mod.
-Homeostasis threshold mod.
-Homeostasis_rate_max_distance

Anymore buffers needed?
*/ 
__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void homeostasis(
    const uint            current_tick,
    const uint            num_neurons,
    const uint            ticks_between_homeostasis,
    const float           homeostasis_weight_mod,
    const float           homeostasis_potentiation_mod,
    const float           homeostasis_depression_mod,
    const float           homeostasis_threshold_mod,
    const float           homeostasis_rate_max_distance,

    __global const uint*  csc_row_ptr,
    __global const uint*  csc_to_csr_index_mapping,

    __global float*       weights,
    __global float*       potentiation_bias,
    __global float*       depression_bias,

    __global const float* spike_rate,
    __global const float* spike_rate_target,
    __global float*       threshold_V,
    const uint            num_synapses,
    const float           fire_rate_tau,        // trace time constant in ticks; computed as ticks_between_homeostasis * 1.5
    __global const uint*  last_update_tick)     // tick of last delivery-kernel run [N]
{
    const uint target_neuron_id = (uint)get_group_id(0);
    const uint lane_id = (uint)get_local_id(0);

    if (target_neuron_id >= num_neurons ||
        ticks_between_homeostasis == 0u ||
        current_tick == 0u ||
        (current_tick % ticks_between_homeostasis) != 0u) {
        return;
    }

    __local uint should_adjust;
    __local float activity_mod;

    if (lane_id == 0u) {
        should_adjust = 0u;
        activity_mod = 1.0f;

        const float target_rate = spike_rate_target[target_neuron_id];
        const float fire_trace  = spike_rate[target_neuron_id];
        // Apply remaining decay for ticks since the delivery kernel last ran this neuron
        const uint  last_tick   = last_update_tick[target_neuron_id];
        const float ticks_since = (current_tick > last_tick) ? (float)(current_tick - last_tick) : 0.0f;
        const float decayed_trace = fire_trace * native_exp(-ticks_since / fire_rate_tau);
        const float firing_rate   = decayed_trace / fire_rate_tau;

        if (target_rate > 0.0f && homeostasis_rate_max_distance > 1.0f) {
            const float mod = firing_rate / target_rate;
            const float distance = (mod >= 1.0f)
                ? mod
                : ((mod > 0.0f) ? (1.0f / mod) : MAXFLOAT);

            if (distance > homeostasis_rate_max_distance) {
                const float min_mod = 1.0f / homeostasis_rate_max_distance;
                should_adjust = 1u;
                activity_mod = clamp(mod, min_mod, homeostasis_rate_max_distance);
            }
        } else if (target_rate <= 0.0f &&
                   decayed_trace > 0.0f &&
                   homeostasis_rate_max_distance > 1.0f) {
            // A zero target rate means any observed spike is over the target
            should_adjust = 1u;
            activity_mod = homeostasis_rate_max_distance;
        }

        if (should_adjust != 0u) {
            const float threshold = threshold_V[target_neuron_id];
            const float threshold_step = clamp(
                (activity_mod - 1.0f) * max(homeostasis_threshold_mod, 0.0f),
                -0.95f,
                0.95f);
            threshold_V[target_neuron_id] +=
                fabs(threshold) * threshold_step;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (should_adjust == 0u) {
        return;
    }

    const uint row_begin = csc_row_ptr[target_neuron_id];
    const uint row_end = csc_row_ptr[target_neuron_id + 1u];

    // Each lane takes care of 1/32 of the incoming connections.
    for (uint csc_index = row_begin + lane_id; csc_index < row_end; csc_index += 32u) {
        const uint csr_index = csc_to_csr_index_mapping[csc_index];
        if (csr_index >= num_synapses) {
            continue;
        }
        const float underactivity_mod = 1.0f / activity_mod;
        const float weight_scale = max(
            0.0f,
            1.0f + ((underactivity_mod - 1.0f) * max(homeostasis_weight_mod, 0.0f)));
        const float potentiation_scale = max(
            0.0f,
            1.0f + ((underactivity_mod - 1.0f) * max(homeostasis_potentiation_mod, 0.0f)));
        const float depression_scale = max(
            0.0f,
            1.0f + ((activity_mod - 1.0f) * max(homeostasis_depression_mod, 0.0f)));

        weights[csr_index] *= weight_scale;
        potentiation_bias[csr_index] *= potentiation_scale;
        depression_bias[csr_index] *= depression_scale;
    }
}
