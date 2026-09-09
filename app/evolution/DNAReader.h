#pragma once

#include "HostOrganismBuffers.h"
#include "HostAppParams.h"
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>


/*
Recieves the DNA genome from organism, and 
method 1:builds CPU version of VRAM data from json DNA genome
method 2:sends it to VRAM Controller (to build GPU version)


neuron type in VRAM:
hidden = 0
input(mouse) then input(error) contiguous; optional leading screen slice size is always 0 here
output = output slots
*/

class DNAReader {
public:
    DNAReader();
    void readDNAFile(const std::string& dna_file_path);
    HostOrganismBuffers getOrganismBuffers() const {
        return h_organism_buffers_;
    }
    HostAppParams getAppParams() const {
        return h_app_params_;
    }
private:
    void createNeurons();
    void assignGlobalParams();
    void assignAppParams();
    void createCSRCSC();
    void createTraceLUT();
    void createPipelineBuffers();

    HostOrganismBuffers h_organism_buffers_;
    HostAppParams h_app_params_;
    nlohmann::json dna_json_;
    std::unordered_map<uint32_t, uint32_t> neuron_index_by_innovation_id_;
};
