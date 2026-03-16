/**
 * @file control_thread.c
 * @brief 1kHz motor control loop example (TIM6 triggered)
 *
 * Demonstrates:
 *   - TIM6 ISR → osThreadFlagsSet → control_task wakeup
 *   - PI current control (simulated plant)
 *   - Multiple udp_print() calls per loop for streaming
 *   - Mutex-protected shared parameters for command interface
 */

#include "control_thread.h"
#include "udp_server.h"
#include "eth_config.h"

#include "cmsis_os2.h"
#include "stm32h7xx_hal.h"

#include <string.h>
#include <math.h>

/* ════════════════════════════════════════════════════════════════
 * Streaming Packet Types (sent via udp_print)
 * ════════════════════════════════════════════════════════════════ */

/** Packet ID for identifying stream data type on PC side */
#define STREAM_ID_CURRENT   0x01
#define STREAM_ID_VOLTAGE   0x02
#define STREAM_ID_DEBUG     0xFF

typedef struct __attribute__((packed)) {
    uint8_t  id;
    uint32_t timestamp_ms;
    float    i_ref;
    float    i_meas;
} stream_current_t;

typedef struct __attribute__((packed)) {
    uint8_t  id;
    uint32_t timestamp_ms;
    float    v_dc;
    float    v_out;
} stream_voltage_t;

typedef struct __attribute__((packed)) {
    uint8_t  id;
    uint32_t timestamp_ms;
    float    error;
    float    integral;
    uint32_t loop_count;
} stream_debug_t;

/* ════════════════════════════════════════════════════════════════
 * Static State
 * ════════════════════════════════════════════════════════════════ */

static control_params_t s_params = {
    .kp          = 1.0f,
    .ki          = 100.0f,
    .current_ref = 0.0f,
};

static osMutexId_t    s_params_mutex   = NULL;
static osThreadId_t   s_control_handle = NULL;

/* PI controller internal state */
static float s_integral     = 0.0f;
static uint32_t s_loop_count = 0;

/* TIM6 handle (defined by CubeMX in main.c, declared extern here) */
extern TIM_HandleTypeDef htim6;

/* ════════════════════════════════════════════════════════════════
 * Thread-safe Parameter Access
 * ════════════════════════════════════════════════════════════════ */

void control_get_params(control_params_t *out)
{
    if (out == NULL) return;
    osMutexAcquire(s_params_mutex, osWaitForever);
    *out = s_params;
    osMutexRelease(s_params_mutex);
}

void control_set_params(const control_params_t *in)
{
    if (in == NULL) return;
    osMutexAcquire(s_params_mutex, osWaitForever);
    s_params.kp          = in->kp;
    s_params.ki          = in->ki;
    s_params.current_ref = in->current_ref;
    osMutexRelease(s_params_mutex);
}

/* ════════════════════════════════════════════════════════════════
 * TIM6 ISR Callback
 *
 * Called at 1kHz by HAL TIM6 interrupt.
 * Wakes the control task via thread flags.
 *
 * NOTE: If your project uses HAL_TIM_PeriodElapsedCallback for
 * multiple timers, add a check: if (htim->Instance == TIM6).
 * ════════════════════════════════════════════════════════════════ */

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance == TIM6) {
        if (s_control_handle != NULL) {
            osThreadFlagsSet(s_control_handle, 0x01);
        }
    }
}

/* ════════════════════════════════════════════════════════════════
 * Control Task (1kHz, osPriorityHigh)
 *
 * Execution time budget: ~0.5ms
 * Remaining ~0.5ms is available for lower-priority tasks.
 * ════════════════════════════════════════════════════════════════ */

static void control_task(void *arg)
{
    (void)arg;

    /* --- Local copies to avoid mutex contention in hot path --- */
    float kp          = s_params.kp;
    float ki          = s_params.ki;
    float current_ref = s_params.current_ref;

    for (;;) {
        /* Wait for TIM6 trigger (1kHz) */
        osThreadFlagsWait(0x01, osFlagsWaitAny, osWaitForever);

        uint32_t tick = HAL_GetTick();

        /* ── 1. Read latest parameters (periodically, not every loop) ── */
        if ((s_loop_count % 10) == 0) {
            osMutexAcquire(s_params_mutex, 0); /* non-blocking try */
            kp          = s_params.kp;
            ki          = s_params.ki;
            current_ref = s_params.current_ref;
            osMutexRelease(s_params_mutex);
        }

        /* ── 2. Sensor Read (simulated) ── */
        /* Replace with actual ADC read in production:
         * float i_meas = read_adc_current();
         */
        float i_meas = current_ref * 0.95f
                       + 0.01f * sinf((float)s_loop_count * 0.1f);

        float v_dc = 24.0f; /* simulated DC bus voltage */

        /* ── 3. PI Control ── */
        float error = current_ref - i_meas;
        s_integral += error * (CONTROL_PERIOD_MS * 0.001f); /* dt = 1ms */

        /* Anti-windup clamp */
        float integral_limit = 10.0f;
        if (s_integral > integral_limit)  s_integral = integral_limit;
        if (s_integral < -integral_limit) s_integral = -integral_limit;

        float v_out = kp * error + ki * s_integral;

        /* Output clamp */
        if (v_out > v_dc)   v_out = v_dc;
        if (v_out < -v_dc)  v_out = -v_dc;

        /* ── 4. PWM Update (simulated) ── */
        /* Replace with actual PWM write in production:
         * __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_1,
         *     (uint32_t)((v_out / v_dc + 1.0f) * 0.5f * htim1.Init.Period));
         */

        /* ── 5. Update shared state ── */
        osMutexAcquire(s_params_mutex, 0);
        s_params.current_meas = i_meas;
        s_params.voltage_out  = v_out;
        s_params.speed_rpm    = 1500.0f; /* placeholder */
        osMutexRelease(s_params_mutex);

        /* ── 6. UDP Streaming (multiple udp_print calls) ── */

        /* Current data — every loop */
        stream_current_t cur_pkt = {
            .id           = STREAM_ID_CURRENT,
            .timestamp_ms = tick,
            .i_ref        = current_ref,
            .i_meas       = i_meas,
        };
        udp_print(&cur_pkt, sizeof(cur_pkt));

        /* Voltage data — every loop */
        stream_voltage_t vol_pkt = {
            .id           = STREAM_ID_VOLTAGE,
            .timestamp_ms = tick,
            .v_dc         = v_dc,
            .v_out        = v_out,
        };
        udp_print(&vol_pkt, sizeof(vol_pkt));

        /* Debug data — every 100th loop (10 Hz) */
        if ((s_loop_count % 100) == 0) {
            stream_debug_t dbg_pkt = {
                .id           = STREAM_ID_DEBUG,
                .timestamp_ms = tick,
                .error        = error,
                .integral     = s_integral,
                .loop_count   = s_loop_count,
            };
            udp_print(&dbg_pkt, sizeof(dbg_pkt));
        }

        s_loop_count++;
    }
}

/* ════════════════════════════════════════════════════════════════
 * Initialization
 * ════════════════════════════════════════════════════════════════ */

void control_thread_init(void)
{
    /* Create mutex for parameter protection */
    const osMutexAttr_t mutex_attr = {
        .name = "ctrl_mtx",
        .attr_bits = osMutexPrioInherit, /* prevent priority inversion */
    };
    s_params_mutex = osMutexNew(&mutex_attr);

    /* Create control task */
    const osThreadAttr_t task_attr = {
        .name       = "control",
        .stack_size = CONTROL_TASK_STACK * 4,
        .priority   = CONTROL_TASK_PRIO,
    };
    s_control_handle = osThreadNew(control_task, NULL, &task_attr);

    /* Start TIM6 interrupt (1kHz trigger) */
    HAL_TIM_Base_Start_IT(&htim6);
}
