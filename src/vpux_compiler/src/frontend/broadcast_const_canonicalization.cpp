//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/frontend/broadcast_const_canonicalization.hpp"

#include "vpux/compiler/utils/logging.hpp"

#include "openvino/core/shape.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"

#include <cstdint>
#include <algorithm>
#include <cstring>
#include <vector>

namespace vpux {

namespace {

constexpr size_t kMaxConstElements = 65536;      // don't blow up memory expanding big constants
constexpr size_t kMaxExpandedElements = 4000000; // cap the expanded constant size (~8MB at f16)

// Right-aligned numpy-style broadcastability: small must be rank <= big rank,
// and every small dim (right-aligned) must be 1 or equal the big dim.
bool isBroadcastableTo(const ov::Shape& small, const ov::Shape& big) {
    if (small.size() > big.size()) {
        return false;
    }
    const auto rankDiff = big.size() - small.size();
    for (size_t i = 0; i < small.size(); ++i) {
        const auto s = small[i];
        const auto b = big[rankDiff + i];
        if (s != 1 && s != b) {
            return false;
        }
    }
    return true;
}

// Expand the constant's raw data to `big` shape with exact broadcast semantics.
// Bit-exact: raw element bytes are copied, element type is preserved.
std::vector<uint8_t> expandConstData(const ov::op::v0::Constant& cnst, const ov::Shape& big) {
    const auto small = cnst.get_output_shape(0);
    const auto elemSize = cnst.get_element_type().size();
    const auto bigSize = ov::shape_size(big);

    // Right-aligned strides of the small shape (in elements); dims of size 1
    // contribute stride 0 so their index maps to 0.
    std::vector<size_t> smallStrides(small.size(), 0);
    size_t acc = 1;
    for (size_t i = small.size(); i-- > 0;) {
        smallStrides[i] = (small[i] == 1) ? 0 : acc;
        if (small[i] != 1) {
            acc *= small[i];
        }
    }
    const auto rankDiff = big.size() - small.size();

    std::vector<uint8_t> out(bigSize * elemSize, 0);
    std::vector<size_t> bigIdx(big.size(), 0);
    const auto* src = static_cast<const uint8_t*>(cnst.get_data_ptr());
    auto* dst = out.data();
    for (size_t lin = 0; lin < bigSize; ++lin) {
        size_t srcOff = 0;
        for (size_t i = 0; i < small.size(); ++i) {
            srcOff += bigIdx[rankDiff + i] * smallStrides[i];
        }
        std::memcpy(dst + lin * elemSize, src + srcOff * elemSize, elemSize);
        // increment multi-index (odometer)
        for (size_t i = big.size(); i-- > 0;) {
            if (++bigIdx[i] < big[i]) {
                break;
            }
            bigIdx[i] = 0;
        }
    }
    return out;
}

bool canonicalizeEltwise(const std::shared_ptr<ov::Node>& node, size_t& numReplacements) {
    if (!ov::is_type<ov::op::v1::Multiply>(node) && !ov::is_type<ov::op::v1::Add>(node) &&
        !ov::is_type<ov::op::v1::Subtract>(node)) {
        return false;
    }
    for (size_t in = 0; in < 2; ++in) {
        // dynamic-shape guard FIRST: get_input_shape() throws to_shape()
        // on dynamic partial shapes (raw-plugin path never statically
        // reshapes before this pass)
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            if (!node->get_input_partial_shape(i).is_static()) {
                return false;
            }
        }
        if (!node->get_output_partial_shape(0).is_static()) {
            return false;
        }
        const auto other = 1 - in;
        const auto& bigShape = node->get_input_shape(other);
        const auto cnst = ov::as_type_ptr<ov::op::v0::Constant>(node->get_input_node_shared_ptr(in));
        if (cnst == nullptr) {
            continue;
        }
        const auto small = cnst->get_output_shape(0);
        const auto smallSize = ov::shape_size(small);
        const auto bigSize = ov::shape_size(bigShape);
        if (smallSize >= bigSize) {
            continue;  // only broadcast (smaller-to-bigger) cases
        }
        if (smallSize > kMaxConstElements || bigSize > kMaxExpandedElements) {
            continue;  // too big to expand
        }
        if (cnst->get_element_type() == ov::element::string || cnst->get_element_type().size() == 0) {
            continue;
        }
        if (!isBroadcastableTo(small, bigShape)) {
            continue;
        }
        auto data = expandConstData(*cnst, bigShape);
        auto expanded = std::make_shared<ov::op::v0::Constant>(cnst->get_element_type(), bigShape, data.data());
        node->input(in).replace_source_output(expanded->output(0));
        ++numReplacements;
        Logger::global().info("BroadcastConstCanonicalization: replacement #%zu: expanded constant %s -> %s on '%s' "
                              "(%s)",
                              numReplacements, small.to_string().c_str(), bigShape.to_string().c_str(),
                              node->get_friendly_name().c_str(), node->get_type_name());
        return true;
    }
    return false;
}

}  // namespace

BroadcastConstCanonicalization::BroadcastConstCanonicalization() : _numReplacements(0) {
    auto m = std::make_shared<ov::pass::pattern::Matcher>(ov::pass::pattern::any_input(),
                                                          "BroadcastConstCanonicalization");
    register_matcher(
            m,
            [this](ov::pass::pattern::Matcher& m) {
                return canonicalizeEltwise(m.get_match_root(), _numReplacements);
            });
}

}  // namespace vpux
