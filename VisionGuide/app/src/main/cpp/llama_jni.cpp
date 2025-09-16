#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include "llama.cpp/include/llama.h"
#include "llama.cpp/src/llama-sampling.h"
#include "llama.cpp/common/common.h"

#define LOG_TAG "LlamaJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Global model and context
static llama_model* model = nullptr;
static llama_context* ctx = nullptr;
static llama_sampler* sampler = nullptr;

const int max_tokens = 512;

// Initialize the sampler with proper parameters
void init_sampler() {
    if (sampler) llama_sampler_free(sampler);

    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    sampler = llama_sampler_chain_init(chain_params);

    // Add temperature sampler
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));

    // Add top-k sampler
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));

    // Add top-p sampler
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1));

    // Add repetition penalty sampler
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
            64,    // n_prev (look back tokens)
            1.1f,  // repeat penalty
            0.0f,  // freq penalty
            0.0f   // present penalty
    ));

    // Add distribution sampler (for final token selection)
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234));

    LOGI("Sampler initialized with chain.");

}

// Tokenizer
std::vector<llama_token> tokenize_prompt(const char *prompt) {
    std::vector<llama_token> tokens(max_tokens);
    int n_tokens = llama_tokenize(
            llama_model_get_vocab(model),
            prompt,
            strlen(prompt),
            tokens.data(),
            max_tokens,
            true,
            false
    );
    tokens.resize(n_tokens);
    LOGI("Tokenized prompt into %d tokens", n_tokens);
    return tokens;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_visionguide_lama_LlamaBridge_00024Companion_initLlamaModel(
        JNIEnv *env, jobject, jstring modelPath_) {

    const char *modelPath = env->GetStringUTFChars(modelPath_, nullptr);

    // Check if file exists and is readable
    FILE* f = fopen(modelPath, "rb");
    if (!f) {
        LOGE("Failed to open model file from path: %s", modelPath);
        env->ReleaseStringUTFChars(modelPath_, modelPath);
        return JNI_FALSE;
    } else {
        fclose(f);
    }

    llama_backend_init();

    llama_log_set([](ggml_log_level level, const char* text, void*) {
        __android_log_print(ANDROID_LOG_ERROR, "LlamaJNI", "llama.cpp: %s", text);
    }, nullptr);

    llama_model_params model_params = llama_model_default_params();
    model = llama_model_load_from_file(modelPath, model_params);
    if (!model) {
        LOGE("Failed to load model from path: %s", modelPath);
        env->ReleaseStringUTFChars(modelPath_, modelPath);
        return JNI_FALSE;
    }

    // Init sampling context
    init_sampler();

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    ctx = llama_init_from_model(model, cp);

    LOGI("LLaMA model loaded successfully, ctx=%p", ctx);
    env->ReleaseStringUTFChars(modelPath_, gmodelPath);
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_visionguide_lama_LlamaBridge_00024Companion_runInference(
        JNIEnv *env, jobject, jstring prompt_, jfloatArray visualTokens_) {

    const char *prompt = env->GetStringUTFChars(prompt_, nullptr);
    LOGI("runInference called with prompt: %s", prompt);

    std::vector<llama_token> text_tokens = tokenize_prompt(prompt);
    LOGI("Number of text tokens: %zu", text_tokens.size());
    env->ReleaseStringUTFChars(prompt_, prompt);

    // Visual tokens
    jsize visual_len = env->GetArrayLength(visualTokens_);
    jfloat *visual_data = env->GetFloatArrayElements(visualTokens_, nullptr);
    int embd_dim = llama_n_embd(model);
    int n_visual_tokens = visual_len / embd_dim;
    LOGI("Visual tokens length: %d, embedding dim: %d, n_visual_tokens: %d", visual_len, embd_dim, n_visual_tokens);

    if (n_visual_tokens <= 0) {
        LOGE("Invalid number of visual tokens, aborting inference.");
        env->ReleaseFloatArrayElements(visualTokens_, visual_data, 0);
        return env->NewStringUTF("");
    }

    // Print first few values for debugging
    for (int i = 0; i < std::min(5, visual_len); ++i) {
        LOGI("Visual token[%d]: %f", i, visual_data[i]);
    }

    llama_batch visual_batch = llama_batch_init(n_visual_tokens, embd_dim, 1);
    visual_batch.n_tokens = n_visual_tokens;
    visual_batch.embd = visual_data;
    visual_batch.token = nullptr;
    visual_batch.pos = (llama_pos*) malloc(n_visual_tokens * sizeof(llama_pos));
    visual_batch.n_seq_id = (int32_t*) malloc(n_visual_tokens * sizeof(int32_t));
    visual_batch.seq_id = (llama_seq_id**) malloc(n_visual_tokens * sizeof(llama_seq_id));
    visual_batch.logits = nullptr;

    static llama_seq_id seq0_val = 0;
    llama_seq_id* seq0_ptr = &seq0_val;

    for (int i = 0; i < n_visual_tokens; ++i) {
        visual_batch.pos[i] = i;
        visual_batch.n_seq_id[i] = 1;
        visual_batch.seq_id[i] = seq0_ptr;
    }

    LOGI("Feeding visual tokens into context...");
    llama_decode(ctx, visual_batch);

    free(visual_batch.pos);
    free(visual_batch.n_seq_id);
    free(visual_batch.seq_id);
    llama_batch_free(visual_batch);
    env->ReleaseFloatArrayElements(visualTokens_, visual_data, 0);

    // Feed text tokens
    llama_batch text_batch = llama_batch_init(text_tokens.size(), 0, 1);
    for (int i = 0; i < text_tokens.size(); ++i) {
        text_batch.token[i] = text_tokens[i];
        text_batch.pos[i] = n_visual_tokens + i;
        text_batch.n_seq_id[i] = 1;
        text_batch.seq_id[i][0] = 0;
        text_batch.logits[i] = false;
    }
    LOGI("Feeding %zu text tokens into context...", text_tokens.size());
    llama_decode(ctx, text_batch);
    llama_batch_free(text_batch);

    for (llama_token tok : text_tokens) llama_sampler_accept(sampler, tok);

    const llama_vocab* vocab = llama_model_get_vocab(model);
    std::vector<llama_token> output_tokens;

    LOGI("Starting token generation...");
    for (int i = 0; i < max_tokens; ++i) {
        llama_token tok = llama_sampler_sample(sampler, ctx, n_visual_tokens + text_tokens.size() + i);
        if (tok == llama_vocab_eos(vocab)) break;

        output_tokens.push_back(tok);
        llama_sampler_accept(sampler, tok);

        if (i < 10) {  // log first few tokens
            char buffer[512];
            llama_token_to_piece(vocab, tok, buffer, sizeof(buffer), 0, false);
            LOGI("Generated token[%d]: %s", i, buffer);
        }

        llama_batch step_batch = llama_batch_init(1, 0, 1);
        common_batch_add(step_batch, tok, n_visual_tokens + text_tokens.size() + i, {0}, false);
        llama_decode(ctx, step_batch);
        llama_batch_free(step_batch);
    }

    LOGI("Token generation completed, total tokens: %zu", output_tokens.size());

    // Convert output to string
    std::string result;
    char buffer[512];
    for (llama_token tok : output_tokens) {
        llama_token_to_piece(vocab, tok, buffer, sizeof(buffer), 0, false);
        result += std::string(buffer);
    }
    LOGI("Generated string length: %zu", result.size());

    return env->NewStringUTF(result.c_str());
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_visionguide_lama_LlamaBridge_00024Companion_cleanup(JNIEnv *env, jobject) {
    if (sampler) llama_sampler_free(sampler);
    if (ctx) llama_free(ctx);
    if (model) llama_model_free(model);
    llama_backend_free();
    LOGI("LLaMA resources cleaned up.");
}
