#pragma once
#include <GigaLearnCPP/Framework.h>
#include <RLGymCPP/BasicTypes/Lists.h>

// Include torch
#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <torch/utils.h>

#define RG_NO_GRAD torch::NoGradGuard _noGradGuard

#define RG_AUTOCAST_ON() { \
at::autocast::set_enabled(true); \
at::autocast::set_autocast_gpu_dtype(torch::kBFloat16); \
at::autocast::set_autocast_cpu_dtype(torch::kFloat); \
}

#define RG_AUTOCAST_OFF() { \
at::autocast::clear_cache(); \
at::autocast::set_enabled(false); \
}

namespace GGL {
	inline torch::ScalarType GetHalfPrecisionType(const torch::Device& device) {
		return device.is_cuda() ? torch::kHalf : torch::kBFloat16;
	}

	template <typename T>
	inline torch::Tensor DIMLIST2_TO_TENSOR(const RLGC::DimList2<T>& list) {
		auto opts = torch::TensorOptions().dtype(torch::CppTypeToScalarType<T>());
		return torch::from_blob(
			(void*)list.data.data(),
			{ (int64_t)list.size[0], (int64_t)list.size[1] },
			opts
		);
	}

	template <typename T>
	inline std::vector<T> TENSOR_TO_VEC(torch::Tensor tensor) {
		assert(tensor.dim() == 1);
		tensor = tensor.contiguous().cpu().detach().to(torch::CppTypeToScalarType<T>());
		T* data = tensor.data_ptr<T>();
		return std::vector<T>(data, data + tensor.size(0));
	}
}