/*
Wrap aggregated atomics on a spiked list
group of 32 performs prefix sum
biggest sum gets representative status and:
        it adds its value to spiked_only_list offser (atomic add)
        it returns the total until that point to use as offset
        it sends offset to other kernels that add its id to the spiked only list att its prefixsum+offset

organization buffers:
-spiked list(becomes prefix sum, then spike only list)
-spiked only list
-spiked only list offset(cl_mem for atomic add)
*/
__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void spiked_only_compaction(
    const uint            num_neurons,
    __global uchar*       spiked_list,
    __global uint*        spiked_only_list,
    __global uint*        spiked_only_list_offset)
{
    const uint neuron_id = (uint)get_global_id(0);
    const uint local_id  = (uint)get_local_id(0);

    __local uint local_prefix[32];
    __local uint group_offset;

    const uint did_spike = (neuron_id < num_neurons && spiked_list[neuron_id] != (uchar)0)
        ? 1u
        : 0u;
    if (neuron_id < num_neurons) {
        spiked_list[neuron_id] = (uchar)0;
    }

    // group of 32 performs prefix sum
    local_prefix[local_id] = did_spike;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint offset = 1u; offset < 32u; offset <<= 1u) {
        const uint addend = (local_id >= offset) ? local_prefix[local_id - offset] : 0u;
        barrier(CLK_LOCAL_MEM_FENCE);
        local_prefix[local_id] += addend;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const uint inclusive_prefix = local_prefix[local_id];
    const uint group_total = local_prefix[31];
    // last neuron in group gets representative status and:
    //         it adds its value to spiked_only_list offser (atomic add)
    //         it returns the total until that point to use as offset
    if (local_id == 31u) {
        group_offset = (group_total > 0u)
            ? atomic_add(spiked_only_list_offset, group_total)
            : 0u;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //kernels that spiked add its id to the spiked only list att its prefixsum+offset
    if (did_spike) {
        const uint rank_in_group = inclusive_prefix - 1u;
        spiked_only_list[group_offset + rank_in_group] = neuron_id;
    }
}
