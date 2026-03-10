#pragma once
#include <RLGymCPP/BasicTypes/Lists.h>

#include "../Util/ModelConfig.h"

namespace GGL {

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/ppo_learner.py
	struct PPOLearnerConfig {

		int64_t tsPerItr = 50'000;
		int64_t batchSize = 50'000;
		int64_t miniBatchSize = 0; // Set to 0 to just use batchSize

		// On the last batch of the iteration, 
		//	if the amount of remaining experience exceeds the batch size, 
		//	all remaining experience is used as a larger batch.
		// This prevents experience loss due to batch size rounding.
		// This will only happen if the amount of remaining experience is < batchSize*2.
		bool overbatching = true;

		// Optional cap for collected timesteps kept in the experience buffer per iteration.
		// Set to 0 to just use tsPerItr.
		int64_t bufferSize = 0;
		int64_t buffer_size = 0;

		double maxEpisodeDuration = 120; // In seconds

		// Actions with the highest probability are always chosen, instead of being more likely
		// This will make your bot play better (usually), but is horrible for learning
		// Trying to run a PPO learn iteration with deterministic mode will throw an exception
		bool deterministic = false;

		// Use half-precision models for inference
		// This is much faster on GPU, not so much for CPU
		bool useHalfPrecision = false;

		// Aliases for useHalfPrecision to make external config code more portable.
		// These fields are synchronized in SyncRuntimeAliases().
		bool useMixedPrecision = false;
		bool use_mixed_precision = false;

		// Recompute parts of the graph during training to reduce peak memory.
		// This trades memory for additional compute.
		bool gradientCheckpointing = false;
		bool gradient_checkpointing = false;

		PartialModelConfig policy, critic, sharedHead;

		int epochs = 2;
		float policyLR = 3e-4f; // Policy learning rate
		float criticLR = 3e-4f; // Critic learning rate

		float entropyScale = 0.018f; // The scale of the normalized entropy loss
		// Whether to ignore invalid actions in the entropy calculation.
		// True means that entropy will be determined only from available actions.
		// False means that entropy for unavailable actions will be zero, 
		//	meaning the entropy of the state is limited to the fraction of available actions in that state.
		bool maskEntropy = false; 

		float clipRange = 0.2f;
		
		// Temperature of the policy's softmax distribution
		float policyTemperature = 1;

		float gaeLambda = 0.95f;
		float gaeGamma = 0.99f;
		float rewardClipRange = 10; // Clip range for normalized rewards, set 0 to disable

		bool useGuidingPolicy = false;
		std::filesystem::path guidingPolicyPath = "guiding_policy/"; // Path of the guiding policy model(s)
		float guidingStrength = 0.03f;

		void SyncRuntimeAliases() {
			bool enabled = useHalfPrecision || useMixedPrecision || use_mixed_precision;
			useHalfPrecision = enabled;
			useMixedPrecision = enabled;
			use_mixed_precision = enabled;

			bool checkpointEnabled = gradientCheckpointing || gradient_checkpointing;
			gradientCheckpointing = checkpointEnabled;
			gradient_checkpointing = checkpointEnabled;

			int64_t resolvedBufferSize = RS_MAX(bufferSize, buffer_size);
			bufferSize = resolvedBufferSize;
			buffer_size = resolvedBufferSize;
		}

		void SyncPrecisionAliases() {
			SyncRuntimeAliases();
		}

		PPOLearnerConfig() {
			policy = {};
			policy.layerSizes = { 256, 256, 256 };
			critic = {};
			critic.layerSizes = { 256, 256, 256 };
			sharedHead = {};
			sharedHead.layerSizes = { 256 };
			sharedHead.addOutputLayer = false;
		}
	};
}