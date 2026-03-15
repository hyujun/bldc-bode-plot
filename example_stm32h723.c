/**
 * @file    example_stm32h723.c
 * @brief   STM32H723 + FreeRTOS usage example for BLDC current controller
 *          bandwidth measurement using chirp, multisine & step excitation.
 *
 * Architecture (FreeRTOS tasks)
 * ─────────────────────────────
 *   signal_task   — high priority, vTaskDelayUntil 1 kHz
 *                   signal generation → FDCAN TX → current read → queue
 *   udp_task      — normal priority, dequeues packets → lwIP UDP TX
 *   main_task     — low priority, measurement control, progress report
 *
 * Design rationale
 * ────────────────
 *   - Signal generation runs in a high-priority task (not ISR) to keep
 *     deterministic 1 kHz timing while allowing FreeRTOS scheduling.
 *   - UDP transmission is decoupled via a queue so that lwIP (which is
 *     NOT ISR-safe in most configurations) runs in its own task context.
 *   - Thread-safe communication uses FreeRTOS queues and task notifications.
 *
 * Hardware
 * ────────
 *   - ETH  : lwIP + netconn/raw UDP → desktop PC (192.168.1.2:55150)
 *   - FDCAN: current reference → motor driver, measured current ← driver
 */

#include "chirp_generator.h"

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

/*--- Replace these with your actual includes ---*/
// #include "stm32h7xx_hal.h"
// #include "cmsis_os2.h"          /* or "FreeRTOS.h" + "task.h" + "queue.h" */
// #include "lwip/udp.h"
// #include "fdcan_driver.h"

/* ════════════════════════════════════════════════════════════
 * FreeRTOS shims  (remove when using real FreeRTOS headers)
 * ════════════════════════════════════════════════════════════ */
#ifndef configMINIMAL_STACK_SIZE
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "semphr.h"
#endif

/* ════════════════════════════════════════════════════════════
 * Platform stubs — replace with your HAL / BSP calls
 * ════════════════════════════════════════════════════════════ */

static void     fdcan_set_current_ref(float i_ref);
static float    fdcan_read_current(void);
static void     udp_send_raw(const void *data, uint16_t len);

/* ════════════════════════════════════════════════════════════
 * Signal type selection
 * ════════════════════════════════════════════════════════════ */

typedef enum {
    SIG_NONE = 0,
    SIG_CHIRP,
    SIG_MULTISINE,
    SIG_STEP,
} sig_type_t;

/* ════════════════════════════════════════════════════════════
 * Shared state  (protected by FreeRTOS primitives)
 * ════════════════════════════════════════════════════════════ */

/* UDP packet: matches Python receiver "<dff" = double + float + float */
#pragma pack(push, 1)
typedef struct {
    double   timestamp_s;    /* 8 bytes */
    float    i_ref;          /* 4 bytes */
    float    i_meas;         /* 4 bytes */
} udp_packet_t;              /* 16 bytes total */
#pragma pack(pop)

/* Task configuration */
#define SIGNAL_TASK_STACK    512      /* words (2 KB) */
#define UDP_TASK_STACK       512
#define MAIN_TASK_STACK      256

#define SIGNAL_TASK_PRIO     (configMAX_PRIORITIES - 1)   /* highest */
#define UDP_TASK_PRIO        (configMAX_PRIORITIES - 3)   /* normal  */
#define MAIN_TASK_PRIO       (tskIDLE_PRIORITY + 1)       /* low     */

#define UDP_QUEUE_LEN        64      /* buffer ~64 ms of packets */

static TaskHandle_t  h_signal_task;
static TaskHandle_t  h_udp_task;
static TaskHandle_t  h_main_task;
static QueueHandle_t h_udp_queue;

/* Measurement state (written by main_task, read by signal_task) */
static volatile sig_type_t  g_active_signal = SIG_NONE;
static volatile bool        g_measuring     = false;

static chirp_state_t        g_chirp;
static multisine_state_t    g_msine;
static step_state_t         g_step;

/* ════════════════════════════════════════════════════════════
 * Measurement control API  (call from main_task)
 * ════════════════════════════════════════════════════════════ */

/**
 * @brief  Start bandwidth measurement with selected signal type.
 * @param  type  SIG_CHIRP or SIG_MULTISINE
 */
static void measurement_start(sig_type_t type)
{
    /* Suspend signal task while reconfiguring */
    vTaskSuspend(h_signal_task);

    switch (type) {
    case SIG_CHIRP:
        chirp_init(&g_chirp);
        break;
    case SIG_MULTISINE:
        multisine_init(&g_msine);
        break;
    case SIG_STEP:
        step_init(&g_step);
        break;
    default:
        vTaskResume(h_signal_task);
        return;
    }

    /* Flush any stale packets in UDP queue */
    xQueueReset(h_udp_queue);

    g_active_signal = type;
    g_measuring     = true;

    vTaskResume(h_signal_task);
}

/**
 * @brief  Abort measurement early.
 */
static void measurement_stop(void)
{
    g_measuring     = false;
    g_active_signal = SIG_NONE;
    fdcan_set_current_ref(0.0f);
}

/**
 * @brief  Block until current measurement completes.
 * @param  timeout_ms  Maximum wait time [ms], 0 = forever
 * @return true if measurement completed, false if timed out
 *
 * Uses task notification from signal_task (zero CPU while waiting).
 */
static bool measurement_wait(uint32_t timeout_ms)
{
    TickType_t ticks = (timeout_ms == 0) ? portMAX_DELAY
                                         : pdMS_TO_TICKS(timeout_ms);
    return ulTaskNotifyTake(pdTRUE, ticks) > 0;
}

/**
 * @brief  Get progress [0.0 – 1.0].
 */
static float measurement_progress(void)
{
    if (!g_measuring) return 0.0f;

    switch (g_active_signal) {
    case SIG_CHIRP:     return chirp_progress(&g_chirp);
    case SIG_MULTISINE: return multisine_progress(&g_msine);
    case SIG_STEP:      return step_progress(&g_step);
    default:            return 0.0f;
    }
}

/* ════════════════════════════════════════════════════════════
 * signal_task  — 1 kHz real-time loop  (highest priority)
 *
 * Uses vTaskDelayUntil for precise periodic execution.
 * Signal generation + FDCAN are fast (<100 µs).
 * UDP packet is enqueued (non-blocking) for udp_task to send.
 * ════════════════════════════════════════════════════════════ */

static void signal_task(void *arg)
{
    (void)arg;
    TickType_t wake_time = xTaskGetTickCount();

    for (;;) {
        vTaskDelayUntil(&wake_time, pdMS_TO_TICKS(1));   /* 1 kHz */

        if (!g_measuring) {
            continue;
        }

        /* ── 1. Generate excitation signal ───────────────── */
        float i_ref   = 0.0f;
        bool  done    = false;
        float elapsed = 0.0f;

        switch (g_active_signal) {
        case SIG_CHIRP:
            i_ref   = chirp_next(&g_chirp);
            elapsed = chirp_elapsed(&g_chirp);
            done    = chirp_is_done(&g_chirp);
            break;

        case SIG_MULTISINE:
            i_ref   = multisine_next(&g_msine);
            elapsed = multisine_elapsed(&g_msine);
            done    = multisine_is_done(&g_msine);
            break;

        case SIG_STEP:
            i_ref   = step_next(&g_step);
            elapsed = step_elapsed(&g_step);
            done    = step_is_done(&g_step);
            break;

        default:
            continue;
        }

        /* ── 2. Send reference to motor driver ───────────── */
        fdcan_set_current_ref(i_ref);

        /* ── 3. Read measured current ────────────────────── */
        float i_meas = fdcan_read_current();

        /* ── 4. Enqueue UDP packet (non-blocking) ────────── */
        udp_packet_t pkt = {
            .timestamp_s = (double)elapsed,
            .i_ref       = i_ref,
            .i_meas      = i_meas,
        };
        xQueueSend(h_udp_queue, &pkt, 0);  /* drop if full */

        /* ── 5. Check completion ─────────────────────────── */
        if (done) {
            fdcan_set_current_ref(0.0f);
            g_measuring     = false;
            g_active_signal = SIG_NONE;

            /* Notify main_task that measurement is done */
            xTaskNotifyGive(h_main_task);
        }
    }
}

/* ════════════════════════════════════════════════════════════
 * udp_task  — dequeue packets and transmit via lwIP
 *
 * Runs at normal priority.  lwIP is called only from this
 * task context (thread-safe with CMSIS-RTOS lwIP port).
 * ════════════════════════════════════════════════════════════ */

static void udp_task(void *arg)
{
    (void)arg;
    udp_packet_t pkt;

    for (;;) {
        /* Block until a packet is available */
        if (xQueueReceive(h_udp_queue, &pkt, portMAX_DELAY) == pdTRUE) {
            udp_send_raw(&pkt, sizeof(pkt));
        }
    }
}

/* ════════════════════════════════════════════════════════════
 * main_task  — measurement orchestrator  (low priority)
 *
 * Runs experiments sequentially, reports progress via UART.
 * Blocks on task notification while waiting for completion
 * (zero CPU usage during measurement — signal_task does work).
 * ════════════════════════════════════════════════════════════ */

static void main_task(void *arg)
{
    (void)arg;

    /* Wait for lwIP / network to be ready */
    vTaskDelay(pdMS_TO_TICKS(2000));

    /* ── Experiment 1: Chirp ─────────────────────────────── */
    /*
     * Desktop side:
     *   python bandwidth_measure.py --signal chirp
     *   → receives UDP, saves bandwidth_raw_chirp.npz
     */
    measurement_start(SIG_CHIRP);
    measurement_wait(0);   /* block until done */

    /* Pause between experiments */
    vTaskDelay(pdMS_TO_TICKS(3000));

    /* ── Experiment 2: Multisine ─────────────────────────── */
    /*
     * Desktop side:
     *   python bandwidth_measure.py --signal multisine
     *   → receives UDP, saves bandwidth_raw_multisine.npz
     */
    measurement_start(SIG_MULTISINE);
    measurement_wait(0);

    /* Pause between experiments */
    vTaskDelay(pdMS_TO_TICKS(3000));

    /* ── Experiment 3: Step response ─────────────────────── */
    /*
     * Desktop side:
     *   python bandwidth_measure.py --signal step
     *   → receives UDP, saves step_response_result.npz
     *   → plots direct time-domain step response
     */
    measurement_start(SIG_STEP);
    measurement_wait(0);

    /* ── Compare on desktop ──────────────────────────────── */
    /*
     *   python bandwidth_measure.py --compare \
     *       bandwidth_raw_chirp.npz bandwidth_raw_multisine.npz
     *   → saves bandwidth_comparison.png
     */

    /* All done — idle */
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

/* ════════════════════════════════════════════════════════════
 * FreeRTOS entry point
 * ════════════════════════════════════════════════════════════ */

int main(void)
{
    /* --- HAL / BSP init --- */
    // HAL_Init();
    // SystemClock_Config();     /* 550 MHz, AXI/AHB clocks */
    // MX_GPIO_Init();
    // MX_FDCAN1_Init();
    // MX_ETH_Init();
    // MX_LWIP_Init();

    /* --- Create FreeRTOS objects --- */
    h_udp_queue = xQueueCreate(UDP_QUEUE_LEN, sizeof(udp_packet_t));

    xTaskCreate(signal_task, "sig",  SIGNAL_TASK_STACK,
                NULL, SIGNAL_TASK_PRIO, &h_signal_task);
    xTaskCreate(udp_task,    "udp",  UDP_TASK_STACK,
                NULL, UDP_TASK_PRIO,    &h_udp_task);
    xTaskCreate(main_task,   "main", MAIN_TASK_STACK,
                NULL, MAIN_TASK_PRIO,   &h_main_task);

    /* signal_task starts suspended — measurement_start() resumes it */
    vTaskSuspend(h_signal_task);

    vTaskStartScheduler();

    /* Should never reach here */
    for (;;) {}
}

/* ════════════════════════════════════════════════════════════
 * Platform stub implementations  (replace with real code)
 * ════════════════════════════════════════════════════════════ */

static void fdcan_set_current_ref(float i_ref)
{
    /* Send i_ref to motor driver via FDCAN
     *
     * Example:
     *   FDCAN_TxHeaderTypeDef tx_hdr = { ... };
     *   uint8_t data[4];
     *   memcpy(data, &i_ref, 4);
     *   HAL_FDCAN_AddMessageToTxFifoQ(&hfdcan1, &tx_hdr, data);
     */
    (void)i_ref;
}

static float fdcan_read_current(void)
{
    /* Read latest measured current from motor driver
     *
     * Typically updated in FDCAN RX FIFO callback:
     *   HAL_FDCAN_GetRxMessage() → memcpy to g_latest_current
     *
     * Example:
     *   extern volatile float g_latest_current;
     *   return g_latest_current;
     */
    return 0.0f;
}

static void udp_send_raw(const void *data, uint16_t len)
{
    /* Send UDP packet to desktop PC (192.168.1.2:55150)
     *
     * Example with lwIP netconn (thread-safe):
     *   static struct netconn *conn = NULL;
     *   if (!conn) {
     *       conn = netconn_new(NETCONN_UDP);
     *       ip_addr_t dest;
     *       IP4_ADDR(&dest, 192, 168, 1, 2);
     *       netconn_connect(conn, &dest, 55150);
     *   }
     *   struct netbuf *buf = netbuf_new();
     *   void *payload = netbuf_alloc(buf, len);
     *   memcpy(payload, data, len);
     *   netconn_send(conn, buf);
     *   netbuf_delete(buf);
     */
    (void)data;
    (void)len;
}
