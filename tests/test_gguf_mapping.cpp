// Test GGUF tensor name mapping
// Verifies that GGUF shortened names map correctly to model struct paths

#include <cstdio>
#include <cstring>
#include <cassert>

// Forward declare the mapping function from gguf_loader.cpp
// We can't include it directly since it's static, so we'll test via a wrapper
namespace leaxer_qwen {
namespace io {
    // We need to expose this for testing - add a test wrapper in gguf_loader.cpp
    const char * test_map_gguf_tensor_name(const char * gguf_name);
}
}

int main() {
    using namespace leaxer_qwen::io;

    printf("Testing GGUF tensor name mapping...\n\n");

    // Test talker layer mappings (tk_l_N_*)
    {
        const char * result = test_map_gguf_tensor_name("tk_l_0_attn_q_proj_weight");
        assert(result != nullptr);
        assert(strcmp(result, "talker.layers[0].attn_q_proj_weight") == 0);
        printf("✓ tk_l_0_attn_q_proj_weight -> %s\n", result);
    }

    {
        const char * result = test_map_gguf_tensor_name("tk_l_27_ffn_down_proj_weight");
        assert(result != nullptr);
        assert(strcmp(result, "talker.layers[27].ffn_down_proj_weight") == 0);
        printf("✓ tk_l_27_ffn_down_proj_weight -> %s\n", result);
    }

    {
        const char * result = test_map_gguf_tensor_name("tk_l_15_in_ln_weight");
        assert(result != nullptr);
        assert(strcmp(result, "talker.layers[15].in_ln_weight") == 0);
        printf("✓ tk_l_15_in_ln_weight -> %s\n", result);
    }

    // Test code predictor layer mappings (talker_cp_l_N_*)
    {
        const char * result = test_map_gguf_tensor_name("talker_cp_l_0_attn_q_proj_weight");
        assert(result != nullptr);
        assert(strcmp(result, "code_predictor.layers[0].attn_q_proj_weight") == 0);
        printf("✓ talker_cp_l_0_attn_q_proj_weight -> %s\n", result);
    }

    {
        const char * result = test_map_gguf_tensor_name("talker_cp_l_4_post_ln_weight");
        assert(result != nullptr);
        assert(strcmp(result, "code_predictor.layers[4].post_ln_weight") == 0);
        printf("✓ talker_cp_l_4_post_ln_weight -> %s\n", result);
    }

    // Test talker top-level fields (talker_model_*)
    {
        const char * result = test_map_gguf_tensor_name("talker_model_emb_weight");
        assert(result != nullptr);
        assert(strcmp(result, "talker.emb_weight") == 0);
        printf("✓ talker_model_emb_weight -> %s\n", result);
    }

    {
        const char * result = test_map_gguf_tensor_name("talker_model_norm_weight");
        assert(result != nullptr);
        assert(strcmp(result, "talker.norm_weight") == 0);
        printf("✓ talker_model_norm_weight -> %s\n", result);
    }

    {
        const char * result = test_map_gguf_tensor_name("talker_model_codec_embedding_weight");
        assert(result != nullptr);
        assert(strcmp(result, "code_predictor.codec_embedding_weight") == 0);
        printf("✓ talker_model_codec_embedding_weight -> %s\n", result);
    }

    // Test code predictor top-level fields
    {
        const char * result = test_map_gguf_tensor_name("talker_code_predictor_norm_weight");
        assert(result != nullptr);
        assert(strcmp(result, "code_predictor.norm_weight") == 0);
        printf("✓ talker_code_predictor_norm_weight -> %s\n", result);
    }

    {
        const char * result = test_map_gguf_tensor_name("talker_code_predictor_output_heads_0_weight");
        assert(result != nullptr);
        assert(strcmp(result, "code_predictor.output_heads[0]") == 0);
        printf("✓ talker_code_predictor_output_heads_0_weight -> %s\n", result);
    }

    {
        const char * result = test_map_gguf_tensor_name("talker_code_predictor_output_heads_15_weight");
        assert(result != nullptr);
        assert(strcmp(result, "code_predictor.output_heads[15]") == 0);
        printf("✓ talker_code_predictor_output_heads_15_weight -> %s\n", result);
    }

    // Test vocoder fields (decoder_*)
    {
        const char * result = test_map_gguf_tensor_name("decoder_codebooks");
        assert(result != nullptr);
        assert(strcmp(result, "vocoder.codebooks") == 0);
        printf("✓ decoder_codebooks -> %s\n", result);
    }

    {
        const char * result = test_map_gguf_tensor_name("decoder_upsample_0_weight");
        assert(result != nullptr);
        assert(strcmp(result, "vocoder.upsample_weights[0]") == 0);
        printf("✓ decoder_upsample_0_weight -> %s\n", result);
    }

    {
        const char * result = test_map_gguf_tensor_name("decoder_upsample_3_alpha");
        assert(result != nullptr);
        assert(strcmp(result, "vocoder.upsample_alphas[3]") == 0);
        printf("✓ decoder_upsample_3_alpha -> %s\n", result);
    }

    // Test unrecognized names
    {
        const char * result = test_map_gguf_tensor_name("unknown_tensor_name");
        assert(result != nullptr);
        assert(strlen(result) == 0);
        printf("✓ unknown_tensor_name -> (empty)\n");
    }

    {
        const char * result = test_map_gguf_tensor_name(nullptr);
        assert(result != nullptr);
        assert(strlen(result) == 0);
        printf("✓ nullptr -> (empty)\n");
    }

    printf("\nAll GGUF tensor name mapping tests passed!\n");
    return 0;
}
