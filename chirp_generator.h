/**
 * @file    chirp_generator.h
 * @brief   Header-only excitation signal generators for BLDC current
 *          controller bandwidth measurement.
 *
 * Provides two signal types:
 *   1. Logarithmic chirp  — continuous frequency sweep
 *   2. Schroeder multisine — sum of cosines at log-spaced frequencies
 *
 * Target:  STM32H723 (Cortex-M7, single-precision FPU)
 *
 * Parameters match bandwidth_measure.py defaults:
 *   f_start = 5 Hz, f_end = 400 Hz, duration = 30 s,
 *   amplitude = 0.3 A, dc_bias = 0.0 A, fs = 1000 Hz
 */

#ifndef CHIRP_GENERATOR_H
#define CHIRP_GENERATOR_H

#include <math.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ════════════════════════════════════════════════════════════
 * Configuration — edit these to match your measurement setup
 * ════════════════════════════════════════════════════════════ */

#ifndef CHIRP_FS
#define CHIRP_FS              1000.0f   /* Sampling rate [Hz] */
#endif

#ifndef CHIRP_F_START
#define CHIRP_F_START         5.0f      /* Sweep start frequency [Hz] */
#endif

#ifndef CHIRP_F_END
#define CHIRP_F_END           400.0f    /* Sweep end frequency [Hz] */
#endif

#ifndef CHIRP_DURATION
#define CHIRP_DURATION        30.0f     /* Sweep duration [s] */
#endif

#ifndef CHIRP_AMPLITUDE
#define CHIRP_AMPLITUDE       0.3f      /* Peak amplitude [A] */
#endif

#ifndef CHIRP_DC_BIAS
#define CHIRP_DC_BIAS         0.0f      /* DC offset [A] */
#endif

#ifndef CHIRP_MAX_CURRENT
#define CHIRP_MAX_CURRENT     1.0f      /* Safety clamp [A] */
#endif

/* ════════════════════════════════════════════════════════════
 * Internal constants (computed from config)
 * ════════════════════════════════════════════════════════════ */

#define CHIRP_DT              (1.0f / CHIRP_FS)
#define CHIRP_TOTAL_SAMPLES   ((uint32_t)(CHIRP_DURATION * CHIRP_FS))
#define CHIRP_TWO_PI          6.2831853071795864f

/* ════════════════════════════════════════════════════════════
 * Generator state
 * ════════════════════════════════════════════════════════════ */

typedef struct {
    /* Precomputed constants */
    float k;           /* f_end / f_start                              */
    float ln_k;        /* ln(k)                                        */
    float beta;        /* 2π * f_start * T / ln(k)  (phase scale)      */
    float exp_rate;    /* ln(k) / T  (exponent rate per second)        */

    /* Runtime state */
    uint32_t sample;   /* Current sample index                         */
    uint32_t total;    /* Total samples in sweep                       */
    bool     done;     /* True after sweep completes                   */
} chirp_state_t;

/* ════════════════════════════════════════════════════════════
 * API
 * ════════════════════════════════════════════════════════════ */

/**
 * @brief  Initialize the chirp generator. Call once before measurement.
 * @param  state  Pointer to chirp state struct
 */
static inline void chirp_init(chirp_state_t *state)
{
    state->k        = CHIRP_F_END / CHIRP_F_START;
    state->ln_k     = logf(state->k);
    state->beta     = CHIRP_TWO_PI * CHIRP_F_START * CHIRP_DURATION / state->ln_k;
    state->exp_rate = state->ln_k / CHIRP_DURATION;

    state->sample = 0;
    state->total  = CHIRP_TOTAL_SAMPLES;
    state->done   = false;
}

/**
 * @brief  Generate the next chirp sample. Call at fs rate (e.g., in timer ISR).
 * @param  state  Pointer to initialized chirp state
 * @return Target current [A], clamped to [-max_current, +max_current].
 *         Returns dc_bias after sweep completes.
 */
static inline float chirp_next(chirp_state_t *state)
{
    if (state->done) {
        return CHIRP_DC_BIAS;
    }

    float t = (float)state->sample * CHIRP_DT;

    /*
     * Logarithmic chirp, phi = -90° (cosine start):
     *   phase(t) = beta * (exp(exp_rate * t) - 1)
     *   x(t)     = amplitude * cos(phase(t)) + dc_bias
     *
     * Equivalent to scipy.signal.chirp(..., method="logarithmic", phi=-90)
     */
    float phase = state->beta * (expf(state->exp_rate * t) - 1.0f);
    float value = CHIRP_AMPLITUDE * cosf(phase) + CHIRP_DC_BIAS;

    /* Safety clamp */
    if (value > CHIRP_MAX_CURRENT)  value = CHIRP_MAX_CURRENT;
    if (value < -CHIRP_MAX_CURRENT) value = -CHIRP_MAX_CURRENT;

    state->sample++;
    if (state->sample >= state->total) {
        state->done = true;
    }

    return value;
}

/**
 * @brief  Check if the chirp sweep has finished.
 * @param  state  Pointer to chirp state
 * @return true if all samples have been generated
 */
static inline bool chirp_is_done(const chirp_state_t *state)
{
    return state->done;
}

/**
 * @brief  Reset the generator to restart the sweep.
 * @param  state  Pointer to chirp state
 */
static inline void chirp_reset(chirp_state_t *state)
{
    state->sample = 0;
    state->done   = false;
}

/**
 * @brief  Get elapsed time [s] since sweep start.
 * @param  state  Pointer to chirp state
 * @return Elapsed time in seconds
 */
static inline float chirp_elapsed(const chirp_state_t *state)
{
    return (float)state->sample * CHIRP_DT;
}

/**
 * @brief  Get progress as a fraction [0.0, 1.0].
 * @param  state  Pointer to chirp state
 * @return Progress ratio
 */
static inline float chirp_progress(const chirp_state_t *state)
{
    return (float)state->sample / (float)state->total;
}


/* ════════════════════════════════════════════════════════════
 * Multisine generator  (Schroeder-phase, log-spaced freqs)
 *
 *   x(t) = (A / √N) · Σ cos(2π·f_k·t + φ_k) + dc_bias
 *
 *   φ_k = −k·(k−1)·π / N   (Schroeder phase, low crest factor)
 *   f_k  logarithmically spaced in [f_start, f_end]
 * ════════════════════════════════════════════════════════════ */

#ifndef MULTISINE_N_FREQS
#define MULTISINE_N_FREQS     60        /* Number of frequency components */
#endif

#define MULTISINE_PI          3.14159265358979f

typedef struct {
    float freqs[MULTISINE_N_FREQS];     /* Log-spaced frequencies [Hz]   */
    float phases[MULTISINE_N_FREQS];    /* Schroeder phases [rad]        */
    float scale;                        /* amplitude / sqrt(N)           */

    uint32_t sample;                    /* Current sample index          */
    uint32_t total;                     /* Total samples                 */
    bool     done;
} multisine_state_t;

/**
 * @brief  Initialize the multisine generator. Call once before measurement.
 * @param  state  Pointer to multisine state struct
 */
static inline void multisine_init(multisine_state_t *state)
{
    const uint32_t N = MULTISINE_N_FREQS;

    /* Log-spaced frequencies: f_k = f_start * (f_end/f_start)^(k/(N-1)) */
    const float log_ratio = logf(CHIRP_F_END / CHIRP_F_START);
    for (uint32_t k = 0; k < N; k++) {
        float frac    = (float)k / (float)(N - 1);
        state->freqs[k]  = CHIRP_F_START * expf(log_ratio * frac);
        /* Schroeder phase: φ_k = -k*(k-1)*π/N */
        state->phases[k] = -(float)k * (float)(k - 1) * MULTISINE_PI / (float)N;
    }

    state->scale  = CHIRP_AMPLITUDE / sqrtf((float)N);
    state->sample = 0;
    state->total  = CHIRP_TOTAL_SAMPLES;
    state->done   = false;
}

/**
 * @brief  Generate the next multisine sample. Call at fs rate.
 * @param  state  Pointer to initialized multisine state
 * @return Target current [A], clamped to [-max_current, +max_current].
 */
static inline float multisine_next(multisine_state_t *state)
{
    if (state->done) {
        return CHIRP_DC_BIAS;
    }

    const float t = (float)state->sample * CHIRP_DT;
    float value = 0.0f;

    for (uint32_t k = 0; k < MULTISINE_N_FREQS; k++) {
        value += cosf(CHIRP_TWO_PI * state->freqs[k] * t + state->phases[k]);
    }
    value = state->scale * value + CHIRP_DC_BIAS;

    /* Safety clamp */
    if (value > CHIRP_MAX_CURRENT)  value = CHIRP_MAX_CURRENT;
    if (value < -CHIRP_MAX_CURRENT) value = -CHIRP_MAX_CURRENT;

    state->sample++;
    if (state->sample >= state->total) {
        state->done = true;
    }

    return value;
}

/**
 * @brief  Check if the multisine sweep has finished.
 */
static inline bool multisine_is_done(const multisine_state_t *state)
{
    return state->done;
}

/**
 * @brief  Reset the multisine generator.
 */
static inline void multisine_reset(multisine_state_t *state)
{
    state->sample = 0;
    state->done   = false;
}

/**
 * @brief  Get elapsed time [s].
 */
static inline float multisine_elapsed(const multisine_state_t *state)
{
    return (float)state->sample * CHIRP_DT;
}

/**
 * @brief  Get progress [0.0, 1.0].
 */
static inline float multisine_progress(const multisine_state_t *state)
{
    return (float)state->sample / (float)state->total;
}


#ifdef __cplusplus
}
#endif

#endif /* CHIRP_GENERATOR_H */

/*
 * ════════════════════════════════════════════════════════════
 * Example usage (STM32H723 timer ISR)
 * ════════════════════════════════════════════════════════════
 *
 *   #include "chirp_generator.h"
 *
 *   static chirp_state_t     chirp;
 *   static multisine_state_t msine;
 *
 *   typedef enum { SIG_CHIRP, SIG_MULTISINE } sig_type_t;
 *   static sig_type_t active_signal = SIG_CHIRP;
 *
 *   void start_measurement(sig_type_t type) {
 *       active_signal = type;
 *       if (type == SIG_CHIRP)     chirp_init(&chirp);
 *       else                       multisine_init(&msine);
 *   }
 *
 *   // Called at 1 kHz by TIM interrupt
 *   void TIMx_IRQHandler(void) {
 *       if (TIM_SR_UIF) {
 *           TIM_SR &= ~TIM_SR_UIF;
 *
 *           float i_ref, elapsed;
 *           bool  done;
 *
 *           if (active_signal == SIG_CHIRP) {
 *               i_ref   = chirp_next(&chirp);
 *               elapsed = chirp_elapsed(&chirp);
 *               done    = chirp_is_done(&chirp);
 *           } else {
 *               i_ref   = multisine_next(&msine);
 *               elapsed = multisine_elapsed(&msine);
 *               done    = multisine_is_done(&msine);
 *           }
 *
 *           if (!done) {
 *               set_current_reference(i_ref);
 *               send_udp_packet(elapsed, i_ref, read_current());
 *           }
 *       }
 *   }
 *
 * ════════════════════════════════════════════════════════════
 * Compile-time configuration override example:
 *
 *   #define CHIRP_FS            2000.0f
 *   #define CHIRP_AMPLITUDE     0.5f
 *   #define CHIRP_DURATION      20.0f
 *   #define MULTISINE_N_FREQS   40
 *   #include "chirp_generator.h"
 * ════════════════════════════════════════════════════════════
 */
