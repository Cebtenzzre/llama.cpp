// Defines sigaction on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "common.h"

#include "console.h"
#include "llama.h"
#include "grammar-parser.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#define SI_DUMP_SEQUENCES_INTERVAL 40
#define CONT_VOCAB_MAX_SIZE_DIFFERENCE  100
#define CONT_VOCAB_CHECK_START_TOKEN_ID 5

static std::atomic<bool> interrupted {false};
static std::atomic<bool> done {false};

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (interrupted) {
            done.store(true);
        } else {
            interrupted.store(true);
        }
    }
}
#endif


static bool check_unsupported(const gpt_params * params) {
    std::string nope;
    const llama_sampling_params & sparams = params->sparams;

    if (params->embedding)
        nope = "embedding";
    else if (!sparams.grammar.empty())
        nope = "grammar"; // Currently broken most likely
    else if (sparams.cfg_scale != 1.0f)
        nope = "cfg_scale";
    else if (!sparams.cfg_negative_prompt.empty())
        nope = "cfg_negative_prompt";
    else if (!params->path_prompt_cache.empty())
        nope = "prompt cache";
    else if (params->escape)
        nope = "prompt escaping";
    else if (params->interactive || params->interactive_first || params->instruct)
        nope = "interactive mode";
    else if (!params->input_prefix.empty() || !params->input_suffix.empty() || params->input_prefix_bos)
        nope = "input prefix or suffix";
    else if (params->hellaswag)
        nope = "hellaswag";
    else if (params->n_keep != 0)
        nope = "keep";
    else if (!params->antiprompt.empty())
        nope = "reverse prompt";
    else if (params->n_parallel != 1)
        nope = "n_parallel";
    if (!nope.empty()) {
        LOG_TEE("%s: error: We don't support %s here.\n", __func__, nope.c_str());
        return false;
    }
    return true;
}


static bool initialize(
    llama_context **ctx_exp_p, llama_context **ctx_ama_p, llama_model **model_exp_p,
    llama_model **model_ama_p, gpt_params & params, std::vector<llama_token> & embd_inp
) {
    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });


    if (params.rope_freq_base != 10000.0) {
        LOG_TEE("%s: warning: changing RoPE frequency base to %g (default 10000.0)\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 1.0) {
        LOG_TEE("%s: warning: scaling RoPE frequency by %g (default 1.0)\n", __func__, params.rope_freq_scale);
    }

    if (params.n_ctx < 8) {
        LOG_TEE("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    LOG_TEE("%s: build = %d (%s)\n", __func__, LLAMA_BUILD_NUMBER, LLAMA_COMMIT);

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    LOG_TEE("%s: seed  = %u\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    LOG("%s: llama backend init\n", __func__);
    llama_backend_init(params.numa);

    // load the model and apply lora adapter, if any
    LOG("%s: load the model and apply lora adapter, if any\n", __func__);

    std::tie(*model_exp_p, *ctx_exp_p) = llama_init_from_gpt_params(params);

    params.model = params.model_draft;
    params.n_gpu_layers = params.n_gpu_layers_draft;
    std::tie(*model_ama_p, *ctx_ama_p) = llama_init_from_gpt_params(params);

    llama_model * model_exp = *model_exp_p;
    llama_model * model_ama = *model_ama_p;
    llama_context * ctx_exp = *ctx_exp_p;
    llama_context * ctx_ama = *ctx_ama_p;

    if (model_exp == NULL || model_ama == NULL) {
        LOG_TEE("%s: error: unable to load model\n", __func__);
        return false;
    }

    const int n_ctx_train_exp = llama_n_ctx_train(model_exp);
    const int n_ctx_train_ama = llama_n_ctx_train(model_ama);
    if (params.n_ctx > n_ctx_train_exp) {
        LOG_TEE("%s: warning: expert model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train_exp, params.n_ctx);
    }
    if (params.n_ctx > n_ctx_train_ama) {
        LOG_TEE("%s: warning: amateur model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train_ama, params.n_ctx);
    }

    // print system information
    {
        LOG_TEE("\n");
        LOG_TEE("system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    {
        const int n_vocab_exp = llama_n_vocab(model_exp);
        const int n_vocab_ama = llama_n_vocab(model_ama);
        const int vocab_diff  = n_vocab_exp > n_vocab_ama
            ? n_vocab_exp - n_vocab_ama
            : n_vocab_ama - n_vocab_exp;

        if (vocab_diff > CONT_VOCAB_MAX_SIZE_DIFFERENCE) {
            fprintf(stderr, "%s: error: amateur model vocab must closely match expert model to use contrastive decoding but ", __func__);
            fprintf(stderr, "expert vocab size %d does not match amateur vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_exp, llama_n_vocab(model_ama), vocab_diff, CONT_VOCAB_MAX_SIZE_DIFFERENCE);
            return 1;
        }

        for (int i = CONT_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_exp, n_vocab_ama); ++i) {
            const char * token_text_exp = llama_token_get_text(model_exp, i);
            const char * token_text_ama = llama_token_get_text(model_ama, i);
            if (std::strcmp(token_text_exp, token_text_ama) != 0) {
                fprintf(stderr, "%s: error: amateur model vocab must match expert model to use speculation but ", __func__);
                fprintf(stderr, "token %d content differs - expert '%s', amateur '%s'\n", i,
                        llama_token_to_piece(ctx_exp, i).c_str(),
                        llama_token_to_piece(ctx_ama, i).c_str());
                return 1;
            }
        }
    }

    const bool add_bos_exp = llama_should_add_bos_token(model_exp);
    LOG("add_bos expert: %d\n", add_bos_exp);

    const bool add_bos_ama = llama_should_add_bos_token(model_ama);
    LOG("add_bos amateur: %d\n", add_bos_ama);

    if (add_bos_exp != add_bos_ama) {
        fprintf(stderr, "%s: error: amateur model add_bos must match expert model to use contrastive decoding but ", __func__);
        fprintf(stderr, "add_bos_amateur = %d while add_bos_exp = %d\n", add_bos_ama, add_bos_exp);
        return 1;
    }

    if (!params.prompt.empty()) {
        LOG("tokenize the prompt\n");
        embd_inp = ::llama_tokenize(ctx_exp, params.prompt, add_bos_exp);
    }

    LOG("prompt: \"%s\"\n", log_tostr(params.prompt));
    LOG("tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_exp, embd_inp).c_str());

    // Should not run without any tokens
    if (embd_inp.empty()) {
        embd_inp.push_back(llama_token_bos(model_exp));
        LOG("input was considered empty and bos was added: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_exp, embd_inp).c_str());
    }

    const int n_ctx = llama_n_ctx(ctx_exp);
    LOG("n_ctx: %d\n", n_ctx);

    if ((int) embd_inp.size() > n_ctx - 4) {
        LOG_TEE("%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }

    if (params.verbose_prompt) {
        LOG_TEE("\n");
        LOG_TEE("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        LOG_TEE("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            LOG_TEE("%6d -> '%s'\n", embd_inp[i], llama_token_to_piece(ctx_exp, embd_inp[i]).c_str());
        }

        LOG_TEE("\n");
    }

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = sigint_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    llama_sampling_params & sparams = params.sparams;
    LOG_TEE("sampling: %s\n", llama_sampling_print(sparams).c_str());
    LOG_TEE("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);
    LOG_TEE("\n\n");

    return true;
}

static bool feed_prompt(
        llama_context *ctx_exp,
        llama_context *ctx_ama,
        const gpt_params & params,
        llama_batch & batch,
        const std::vector<llama_token> & tokens) {
    int32_t tokens_remain           = tokens.size();
    const llama_token * tokens_curr = tokens.data();
    llama_pos pos = 0;

    console::set_display(console::prompt);
    while (tokens_remain > 0 && !interrupted) {
        const int32_t chunk_size = std::min(int32_t(tokens_remain), params.n_batch);
        llama_batch_clear(batch);
        for (int i = 0; i < chunk_size; i++) {
            llama_batch_add(batch, tokens_curr[i], pos + i, {0}, false);
        }
        pos += batch.n_tokens;
        tokens_remain -= batch.n_tokens;
        batch.logits[batch.n_tokens - 1] = tokens_remain < 1;

        if (llama_decode(ctx_exp, batch) != 0) {
            console::set_display(console::reset);
            LOG_TEE("%s : failed to eval expert\n", __func__);
            return false;
        }
        if (llama_decode(ctx_ama, batch) != 0) {
            console::set_display(console::reset);
            LOG_TEE("%s : failed to eval amateur\n", __func__);
            return false;
        }

        // display text
        for (int i = 0; i < batch.n_tokens; i++) {
            const std::string token_str = llama_token_to_piece(ctx_exp, tokens_curr[i]);
            fputs(token_str.c_str(), stdout);
        }
        fflush(stdout);

        tokens_curr += batch.n_tokens;
    }
    console::set_display(console::reset);
    return true;
}


int main(int argc, char ** argv) {
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (!check_unsupported(&params)) {
        return 1;
    }

    if (params.model_draft.empty()) {
        fprintf(stderr, "%s: error: --model-draft is required\n", __func__);
        return 1;
    }

#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("simple-inference", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc,argv);
#endif // LOG_DISABLE_LOGS

    llama_context * ctx_exp   = NULL;
    llama_context * ctx_ama   = NULL;
    llama_model *   model_exp = NULL;
    llama_model *   model_ama = NULL;
    std::vector<llama_token> prompt_tokens;

    if (!initialize(&ctx_exp, &ctx_ama, &model_exp, &model_ama, params, prompt_tokens)) {
        return 1;
    }

    const int n_ctx = llama_n_ctx(ctx_exp);
    int n_remain    = params.n_predict;

    const size_t prompt_size = prompt_tokens.size();
    llama_batch batch = llama_batch_init(std::max(int32_t(prompt_size), params.n_batch), 0, 1);
    llama_sampling_context * ctx_sampling = llama_sampling_init(params.sparams);

    if (!feed_prompt(ctx_exp, ctx_ama, params, batch, prompt_tokens)) {
        return 1;
    }

    if (n_remain < 0 || n_remain + int(prompt_size) > n_ctx) {
        n_remain = n_ctx - prompt_size;
    }

    size_t n_past = prompt_size;

    size_t decode_count = 0;
    int64_t decode_time_total = 0, decode_time_last = 0;

    const int n_vocab_exp = llama_n_vocab(model_exp);
    const int n_vocab = std::min(n_vocab_exp, llama_n_vocab(model_ama));

    const float cd_alpha = 0.1; // TODO: make CLI argument
    const float cd_beta = 0.5; // set to 0.5 to behave like original paper
    const float mask = std::numeric_limits<float>::lowest();

    while (!interrupted) {
        int idx = batch.n_tokens - 1;
        float * logits_exp = llama_get_logits_ith(ctx_exp, idx);
        float * logits_ama = llama_get_logits_ith(ctx_ama, idx);

#if 1
        float max_logit_exp = *std::max_element(logits_exp, logits_exp + n_vocab);

        for (int i = 0; i < n_vocab_exp; ++i) {
            // NB: original paper applies alpha to probabilities, further paper defines in terms of log probs
            //     both have the same meaning
            if (logits_exp[i] < max_logit_exp + log(cd_alpha)) {
                // not a plausible token according to expert
                logits_exp[i] = mask;
            } else if (i >= n_vocab) {
                // token not known to amateur
                logits_exp[i] = mask;
            } else {
                logits_exp[i] = (1 + cd_beta) * logits_exp[i] - cd_beta * logits_ama[i];
            }
        }
#endif

        const llama_token id = llama_sampling_sample(ctx_sampling, ctx_exp, NULL, idx);
        llama_sampling_accept(ctx_sampling, ctx_exp, id, true);

        n_past++;
        n_remain--;

        // end of text token
        if (id == llama_token_eos(model_exp) || n_remain == 0) {
            LOG_TEE(" [end of text]\n");
            break;
        }

        batch.n_tokens = 0;
        llama_batch_add(batch, id, n_past, {0}, true);

        const std::string token_str = llama_token_to_piece(ctx_exp, id);
        fputs(token_str.c_str(), stdout);
        fflush(stdout);

        decode_time_last = ggml_time_us();
        int decode_result_exp = llama_decode(ctx_exp, batch);
        int decode_result_ama = llama_decode(ctx_ama, batch);
        decode_time_last = std::max(int64_t(0), ggml_time_us() - decode_time_last);
        decode_time_total += decode_time_last;

        if (decode_result_exp != 0) {
            LOG_TEE("%s : failed to eval batch of size %d: %s\n", __func__, batch.n_tokens,
                decode_result_exp == 1 ? "couldn't find slot" : "unknown error");
            return 1;
        }
        if (decode_result_ama != 0) {
            LOG_TEE("%s : failed to eval batch of size %d: %s\n", __func__, batch.n_tokens,
                decode_result_ama == 1 ? "couldn't find slot" : "unknown error");
            return 1;
        }
        decode_count++;
    }

    puts("");
    console::cleanup();

    LOG_TEE("\namateur:\n");
    llama_print_timings(ctx_ama);

    LOG_TEE("\nexpert:\n");
    llama_print_timings(ctx_exp);

    llama_sampling_free(ctx_sampling);

    llama_batch_free(batch);

    llama_free(ctx_exp);
    llama_free(ctx_ama);
    llama_free_model(model_exp);
    llama_free_model(model_ama);

    llama_backend_free();

#ifndef LOG_DISABLE_LOGS
    LOG_TEE("Log end\n");
#endif // LOG_DISABLE_LOGS

    return interrupted ? 130 : 0;
}
