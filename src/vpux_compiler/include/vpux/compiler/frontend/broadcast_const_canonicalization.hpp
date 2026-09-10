//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace vpux {

/**
 * @brief Expands small broadcast constant operands of elementwise ops to the
 *        other operand's static shape (exact broadcast / tile, bit-exact data
 *        copy, element type preserved).
 *
 * The NPU3720 DPU broadcast eltwise path for tiny constants (e.g. RMSNorm
 * gamma multiplies Multiply([1,1,1536], Constant[1,1,1])) runs at ~0.3
 * elem/us while a dense same-shape multiply runs at DPU rates. This pass
 * canonicalizes the broadcast away at the nGraph level, before the IE
 * dialect pipeline, by physically expanding the constant.
 *
 * Guards: constant must be strictly smaller than the other operand, both
 * shapes static, constant <= 64K elements and expanded size <= 4M elements.
 */
class BroadcastConstCanonicalization : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("BroadcastConstCanonicalization");
    BroadcastConstCanonicalization();

private:
    size_t _numReplacements;
};

}  // namespace vpux
