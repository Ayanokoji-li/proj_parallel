#pragma once
#include "ef_layout.h"
#include "ef_util.h"
#include "ef_graph_for_pg.cuh"

template <size_t kSkipQuantum, size_t kForwardQuantum>
class EFGraph_ForPG;

template <size_t kSkipQuantum, size_t kForwardQuantum>
class EFLayout_ForPG : public EFLayout
{
    std::vector<double> efVal;
    friend class EFGraph_ForPG<kSkipQuantum, kForwardQuantum>;

public:
    EFLayout_ForPG(const csr_for_pg& csr) : EFLayout(csr) {efVal = csr.csrVal;};

    const std::vector<double>& getEFVal() const {return efVal;};
};