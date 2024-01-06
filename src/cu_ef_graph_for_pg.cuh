#pragma once
#include "cuda_util.cuh"
#include "ef_graph_for_pg.cuh"
#include "ef_param.h"

template <size_t kSkipQuantum, size_t kForwardQuantum>
class CUEFGraph_ForPG : public CUEFGraph
{
private:
    const double* d_Val;

public:
    CUEFGraph_ForPG(const EFGraph_ForPG<kSkipQuantum, kForwardQuantum>& ef_graph, int alloc_mode = 0)
        : CUEFGraph(ef_graph, alloc_mode), d_Val(ef_graph.GetVal())
    {
    }

    CUEFGraph_ForPG(const EFLayout_ForPG<kSkipQuantum, kForwardQuantum>& ef_layout, int alloc_mode = 0)
        : CUEFGraph(ef_layout, alloc_mode), d_Val(ef_layout.GetVal())
    {
    }

    __device__ __forceinline__ double GetVal(size_t i) const
    {
        return d_Val[i];
    }
}