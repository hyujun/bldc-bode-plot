/**
 * @file control_thread.h
 * @brief 1kHz motor control thread (TIM6 triggered)
 *
 * Example control loop that demonstrates integration with
 * the UDP streaming (udp_print) and command-response system.
 * TIM6 generates 1kHz interrupts to wake the control task.
 */

#ifndef CONTROL_THREAD_H
#define CONTROL_THREAD_H

#include <stdint.h>

/* ════════════════════════════════════════════════════════════════
 * Control Parameters (shared with UDP command handler)
 * ════════════════════════════════════════════════════════════════ */

typedef struct {
    /* Writable via command (kp, ki, current_ref) */
    float kp;               /**< Proportional gain */
    float ki;               /**< Integral gain */
    float current_ref;      /**< Current setpoint [A] */

    /* Read-only status (updated by control loop) */
    float current_meas;     /**< Measured current [A] */
    float voltage_out;      /**< Controller output voltage [V] */
    float speed_rpm;        /**< Estimated speed [RPM] */
} control_params_t;

/**
 * @brief Get a thread-safe snapshot of control parameters.
 * @param out  Destination struct
 */
void control_get_params(control_params_t *out);

/**
 * @brief Set writable control parameters (kp, ki, current_ref).
 *
 * Only kp, ki, and current_ref are written; read-only fields
 * in the input struct are ignored.
 *
 * @param in  Source struct
 */
void control_set_params(const control_params_t *in);

/**
 * @brief Initialize control thread.
 *
 * - Creates mutex for parameter protection
 * - Creates the control_task (osPriorityHigh)
 * - Starts TIM6 in interrupt mode (1kHz)
 *
 * Call after MX_TIM6_Init() and udp_server_init().
 */
void control_thread_init(void);

#endif /* CONTROL_THREAD_H */
