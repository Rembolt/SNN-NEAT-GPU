
/*
Read from updated_list_buffer dirty bits N/32
dt_ms = (eg 0.1f= 0.1ms per tick) fixed global constant 
then check individually for update (quit if none true)
Utilize AdEx to apply decay and add charges from delay_circular_buffer
update adEx formula from last updated tick to this tick.
    neuron_id  = <from dirty bit scan>
    dt = current_tick - last_updated_tick[neuron_id]
    bool has_spiked = false

    for (step = 0 to dt - 1)
        sim_tick = last_updated_tick[neuron_id] + step
        if (sim_tick < refractory_end)
            v = v_rest
            float dw = (a_param * (v_rest - E_Leak) - w_curr) / tau_w
            w_curr += dw * dt_ms
            continue
        float exp_term = exp((v - V_peak) / delta_T)
        float dv = (-g_Leak * (v - E_Leak) + g_Leak * delta_T * exp_term - w_curr) / C
        float dw = (a_param * (v - E_Leak) - w_curr) / tau_w
        v += dv * dt_ms
        w_curr += dw * dt_ms
        if (v >= v_threshold)
            v = v_rest
            w_curr += b_param
            ref_end = sim_tick + refractory_period_length
            has_spiked = true
            break

    if (current_tick >= ref_end)
        v += incoming_delay_charge_buffer[neuron_id]
        if (v >= v_threshold)
            v = v_rest
            w_curr += b_param
            ref_end = current_tick + refractory_period_length
            has_spiked = true
    incoming_delay_charge_buffer[neuron_id] = 0.0f
    V[neuron_id] = v
    w[neuron_id] = w_curr
    refractory_end_tick[neuron_id] = ref_end
    last_update_tick[neuron_id] = current_tick
    if (has_spiked) spiked_list[neuron_id] = 1
    
check if refractory period ended write new potential to neuron
threshhold check (if fired) reset potential
                            add to spiked list
                            last_fired_tick update
                            reset refractory period
                            spike count++
                            other updates

per neuron 1d buffers:
-Potential current charge (V)
-Adaptation current (w)
-Subthreshold adaptation strength (a)
-Post Spike-Triggered (w) increment (b)
-Adaptation time constant (t_w)
-threshold (V_Threshold)
-Resting potential (V_rest)
-Last decay last updated tick
-Refractory end tick
-Refractory length
-trace last update

global params buffers:
-Current tick
-refractory period length
-Membrane Capacitance (C)
-Leak conductance (g_Leak)
-Leak reversal potential (resting) (E_Leak)
-Slope factor (spike sharpness) (delta_T)
-Spike peak voltage (V_peak)

organization buffers:
-current tick
-circular updated list dirty bit(2d, N/32 | max delay)(who recieved charge)
-spiked list (on off)
-circular delay list(2d, column size = max delay)

Anymore buffers needed?
*/ 

__kernel void delivery_decay_fire(

    //Global params
    const uint current_tick,
    const uint N,                    // total neuron count
    const uint words_per_row,        // ceil(N / 32)
    const uint max_delay,            // circular buffer depth (ticks)
    const float dt_ms,                // milliseconds per tick  (e.g. 0.1f)
    const float C,                    // membrane capacitance
    const float g_Leak,               // leak conductance
    const float E_Leak,               // leak reversal potential
    const float delta_T,              // slope factor (spike sharpness)
    const float V_peak,               // soft spike peak (exp term reference)

    //Per-neuron
    __global float* V,                        // membrane potential      [N]
    __global float* w,                        // adaptation current      [N]
    __global uint*        last_update_tick,      // last tick this neuron ran    [N]
    __global uint*        refractory_end_tick,   // tick refractoriness expires  [N]
    __global const uint*  refractory_period_len, // per-neuron refractory length [N]

    //genome params
    __global const float* a_param,            // sub-threshold coupling  [N]
    __global const float* b_param,            // spike adaptation jump   [N]
    __global const float* tau_w,              // adaptation time const   [N]
    __global const float* v_threshold,        // spike threshold         [N]
    __global const float* v_rest,             // reset/resting potential [N]
    __global float*       spike_rate,         // leaky firing-rate trace [N]; decays exp(-1/fire_rate_tau) each tick

    //Circular delivery buffers
    //indexed by [delay_slot * stride + neuron_id]
    // dirty_bits:     stride = words_per_row,  unit = packed uint32
    //                 bit (neuron_id % 32) of word (neuron_id / 32)
    // delay_charges:  stride = N,              unit = float
    __global uint*        dirty_bits,            // [max_delay * words_per_row]
    __global float*       delay_charges,         // [max_delay * N]

    //Outputs
    __global uchar*       spiked_list,           // 1 = spiked this tick  [N] (host pre-zeroes)
    __global uint*        last_fired_tick,        // tick of last spike     [N] (STDP pairing)
    const float           fire_rate_tau           // trace time constant in ticks; computed as ticks_between_homeostasis * 1.5
)
{
    const int neuron_id = (int)get_global_id(0);
    if (neuron_id >= N) return;   // guard non-power-of-2 launches
    if (max_delay == 0u) return;

    const uint tick_now   = current_tick;
    const uint delay_slot = tick_now % max_delay;

    //Dirty-bit check — early exit if no charge
    //Each row of dirty_bits has words_per_row uint32 words
    const int  word_index  = neuron_id >> 5;          // neuron_id / 32
    const int  bit_index   = neuron_id & 31;           // neuron_id % 32
    const int dirty_index = (int)(delay_slot * words_per_row) + word_index;
    const uint dirty_mask = 1u << bit_index;
    const uint dirty_word = dirty_bits[dirty_index];
    const int charge_index = (int)(delay_slot * N) + neuron_id;

    if ((dirty_word & dirty_mask) == 0u) {
        delay_charges[charge_index] = 0.0f;
        return;
    }
    atomic_and(&dirty_bits[dirty_index], ~dirty_mask);

    //Load
    float v_current       = V[neuron_id];
    float w_current       = w[neuron_id];
    int   last_update     = (int)last_update_tick[neuron_id];
    uint  refractory_end  = refractory_end_tick[neuron_id];
    float fire_trace      = spike_rate[neuron_id];

    const float subthreshold_adaptation  = a_param[neuron_id];
    const float spike_adaptation_jump    = b_param[neuron_id];
    const float adaptation_time_constant = tau_w[neuron_id];     // assumed > 0
    const float spike_threshold          = v_threshold[neuron_id];
    const float resting_potential        = v_rest[neuron_id];
    const uint  refractory_length        = refractory_period_len[neuron_id];

    const float dt_ms_value  = dt_ms;
    const float capacitance = C;                                  // assumed > 0
    const float g_leak      = g_Leak;
    const float e_leak      = E_Leak;
    const float slope_delta_t = delta_T;                          // assumed > 0
    const float v_peak      = V_peak;

    //AdEx catch-up loop: [last_update, current_tick)
    // Multiple spike/refractory cycles within the window are handled
    // spike sets refractory_end
    // w must be evolved to current_tick
    const int delta_ticks     = (int)tick_now - last_update;
    const float decay_per_step = native_exp(-1.0f / fire_rate_tau);
    bool      has_spiked    = false;
    int       last_spike_tick = (int)last_fired_tick[neuron_id];  // carry forward for STDP

    for (int step = 0; step < delta_ticks; ++step) {

        const int simulation_tick = last_update + step;
        fire_trace *= decay_per_step;

        // v is clamped to resting_potential
        // w continues to decay
        if ((uint)simulation_tick < refractory_end) {
            v_current = resting_potential;
            float delta_w = (subthreshold_adaptation * (resting_potential - e_leak)
                             - w_current) / adaptation_time_constant;
            w_current += delta_w * dt_ms_value;
            continue;
        }

        //AdEx sub-threshold dynamics
        // exp((v − V_peak) / delta_T):
        //   Clamp argument to [-20, +20] before native_exp
        //   +20 → exp ≈ 485M, safely in float range
        //   −20 → exp ≈ 2e-9, term is negligible and avoids subnormal
        float exponential_argument = clamp((v_current - v_peak) / slope_delta_t,
                                           -20.0f, 20.0f);
        float exponential_term     = native_exp(exponential_argument);

        float delta_v = (  -g_leak * (v_current - e_leak)
                         + g_leak * slope_delta_t * exponential_term
                         - w_current
                        ) / capacitance;

        float delta_w = (subthreshold_adaptation * (v_current - e_leak)
                         - w_current) / adaptation_time_constant;

        v_current += delta_v * dt_ms_value;
        w_current += delta_w * dt_ms_value;

            //Threshold check
        if (v_current >= spike_threshold) {
            v_current       = resting_potential;
            w_current      += spike_adaptation_jump;
            refractory_end  = (uint)simulation_tick + refractory_length;
            has_spiked      = true;
            last_spike_tick = simulation_tick;
            fire_trace     += 1.0f;
            // loop continues to catch-up
        }
    }

    //Incoming charge delivery
    const float incoming_charge = delay_charges[charge_index];
    delay_charges[charge_index] = 0.0f;                          // unconditional zero

    if (tick_now >= refractory_end) {

        if (incoming_charge != 0.0f) {
            v_current += incoming_charge;

            if (v_current >= spike_threshold) {
                v_current       = resting_potential;
                w_current      += spike_adaptation_jump;
                refractory_end  = (uint)tick_now + refractory_length;
                has_spiked      = true;
                last_spike_tick = (int)tick_now;                  // charge lands at current_tick
                fire_trace     += 1.0f;
            }
        }
    }

    //Write back
    V[neuron_id]                   = v_current;
    w[neuron_id]                   = w_current;
    refractory_end_tick[neuron_id] = refractory_end;
    last_update_tick[neuron_id]    = tick_now;
    spike_rate[neuron_id]          = fire_trace;

    if (has_spiked) {
        spiked_list[neuron_id]     = 1;
        last_fired_tick[neuron_id] = (uint)last_spike_tick;
    }
}
