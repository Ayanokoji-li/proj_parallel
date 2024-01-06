#pragma once
#include "ef_graph.h"
#include "ef_util.h"
#include "ef_layout.h"
#include "ef_layout_for_pg.h"

template <size_t kSkipQuantum, size_t kForwardQuantum>
class EFLayout_ForPG;

template <size_t kSkipQuantum, size_t kForwardQuantum>
class EFGraph_ForPG : public EFGraph
{
    std::vector<double> efgVal;

public:
    
    EFGraph_ForPG(const EFLayout_ForPG<kSkipQuantum, kForwardQuantum>& ef_layout_for_pg)
    {
        efgVal = ef_layout_for_pg.efVal;
    }
}