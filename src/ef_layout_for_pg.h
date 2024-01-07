#pragma once
#include "ef_layout.h"
#include "ef_util.h"
#include "ef_graph_for_pg.cuh"

template <size_t kSkipQuantum, size_t kForwardQuantum>
class EFGraph_ForPG;

template <size_t kSkipQuantum, size_t kForwardQuantum>
class EFLayout_ForPG : public EFLayout<kSkipQuantum, kForwardQuantum>
{
    std::vector<double> efVal;
    friend class EFGraph_ForPG<kSkipQuantum, kForwardQuantum>;

public:
    EFLayout_ForPG(const CSR_for_pg& csr) : EFLayout<kSkipQuantum, kForwardQuantum>(csr) {efVal = csr.csrVal;};

    const std::vector<double>& GetVal() const {return efVal;};
};