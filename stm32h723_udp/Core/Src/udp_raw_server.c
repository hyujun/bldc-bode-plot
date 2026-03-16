/**
 * @file udp_raw_server.c
 * @brief High-speed UDP server using LwIP Raw API (callback-based)
 *
 * Port 51550: TX-only streaming — ring buffer drained by sys_timeout timer
 * Port 51551: Ultra-low-latency command-response — recv callback, zero task switch
 *
 * All Raw API calls execute in tcpip_thread context.
 * No FreeRTOS tasks are created — saves ~4KB stack vs Netconn version.
 */

#include "udp_raw_server.h"
#include "eth_config.h"

#include "lwip/udp.h"
#include "lwip/pbuf.h"
#include "lwip/ip_addr.h"
#include "lwip/sys.h"
#include "lwip/tcpip.h"

#include <string.h>
#include <stdio.h>
#include <stdarg.h>

/* ════════════════════════════════════════════════════════════════
 * Lock-free SPSC Ring Buffer
 *
 * Identical to the Netconn version.
 * Single producer (any context via udp_raw_print) →
 * Single consumer (tcpip_thread via stream drain timer).
 *
 * Each entry: [uint16_t length][payload bytes]
 * ════════════════════════════════════════════════════════════════ */

typedef struct {
    uint8_t           buf[UDP_RING_BUF_SIZE];
    volatile uint32_t head;   /* write position (producer only) */
    volatile uint32_t tail;   /* read  position (consumer only) */
} ring_buf_t;

static ring_buf_t s_ring = {.head = 0, .tail = 0};

/** Available bytes for reading */
static inline uint32_t ring_readable(const ring_buf_t *rb)
{
    uint32_t h = rb->head;
    uint32_t t = rb->tail;
    return (h >= t) ? (h - t) : (UDP_RING_BUF_SIZE - t + h);
}

/** Available space for writing */
static inline uint32_t ring_writable(const ring_buf_t *rb)
{
    return UDP_RING_BUF_SIZE - 1 - ring_readable(rb);
}

/**
 * @brief Write [len(2B) | data(len B)] into ring buffer.
 * @return 0 on success, -1 on overflow (data dropped)
 */
static int ring_write(ring_buf_t *rb, const void *data, uint16_t len)
{
    uint32_t total = (uint32_t)len + 2u;
    if (ring_writable(rb) < total) {
        return -1;
    }

    uint32_t h = rb->head;

    /* Write length prefix (little-endian) */
    rb->buf[h] = (uint8_t)(len & 0xFF);
    h = (h + 1) % UDP_RING_BUF_SIZE;
    rb->buf[h] = (uint8_t)(len >> 8);
    h = (h + 1) % UDP_RING_BUF_SIZE;

    /* Write payload */
    const uint8_t *src = (const uint8_t *)data;
    for (uint32_t i = 0; i < len; i++) {
        rb->buf[h] = src[i];
        h = (h + 1) % UDP_RING_BUF_SIZE;
    }

    __DMB();
    rb->head = h;
    return 0;
}

/**
 * @brief Read one entry from ring buffer.
 * @return Actual payload length, 0 if empty
 */
static uint16_t ring_read(ring_buf_t *rb, void *out, uint16_t max_len)
{
    if (ring_readable(rb) < 2) {
        return 0;
    }

    uint32_t t = rb->tail;

    uint16_t len = rb->buf[t];
    t = (t + 1) % UDP_RING_BUF_SIZE;
    len |= (uint16_t)rb->buf[t] << 8;
    t = (t + 1) % UDP_RING_BUF_SIZE;

    if (len == 0 || ring_readable(rb) < (uint32_t)(len + 2)) {
        return 0;
    }

    uint8_t *dst = (uint8_t *)out;
    uint16_t copy_len = (len <= max_len) ? len : max_len;

    for (uint16_t i = 0; i < len; i++) {
        if (i < copy_len) {
            dst[i] = rb->buf[t];
        }
        t = (t + 1) % UDP_RING_BUF_SIZE;
    }

    __DMB();
    rb->tail = t;
    return copy_len;
}

/* ════════════════════════════════════════════════════════════════
 * Static State
 * ════════════════════════════════════════════════════════════════ */

static struct udp_pcb      *s_stream_pcb  = NULL;  /* Port 51550 TX */
static struct udp_pcb      *s_cmd_pcb     = NULL;  /* Port 51551 RX/TX */
static udp_raw_cmd_handler_t s_cmd_handler = NULL;
static udp_raw_stats_t      s_stats       = {0};

/* ════════════════════════════════════════════════════════════════
 * Public API: udp_raw_print()
 * ════════════════════════════════════════════════════════════════ */

void udp_raw_print(const void *data, uint16_t len)
{
    if (data == NULL || len == 0 || len > UDP_STREAM_MAX_PKT) {
        return;
    }

    if (ring_write(&s_ring, data, len) != 0) {
        s_stats.stream_overflow++;
    }
}

/* ════════════════════════════════════════════════════════════════
 * Public API: udp_raw_printf()
 * ════════════════════════════════════════════════════════════════ */

int udp_raw_printf(const char *fmt, ...)
{
    if (fmt == NULL) {
        return -1;
    }

    char buf[UDP_STREAM_MAX_PKT];
    va_list args;
    va_start(args, fmt);
    int len = vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    if (len < 0) {
        return -1;
    }

    uint16_t send_len = (len < (int)sizeof(buf)) ? (uint16_t)len : (uint16_t)(sizeof(buf) - 1);

    udp_raw_print(buf, send_len);
    return (int)send_len;
}

/* ════════════════════════════════════════════════════════════════
 * Stream Drain Timer (Port 51550 TX)
 *
 * Replaces the Netconn stream task. Called periodically by
 * sys_timeout in tcpip_thread context — no task stack needed.
 * ════════════════════════════════════════════════════════════════ */

static void stream_drain_timer(void *arg)
{
    (void)arg;

    static uint8_t tx_buf[UDP_STREAM_MAX_PKT];
    uint32_t sent = 0;

    while (sent < UDP_STREAM_DRAIN_MAX) {
        uint16_t len = ring_read(&s_ring, tx_buf, sizeof(tx_buf));
        if (len == 0) {
            break;
        }

        /* Allocate pbuf from transport layer pool */
        struct pbuf *p = pbuf_alloc(PBUF_TRANSPORT, len, PBUF_RAM);
        if (p == NULL) {
            break; /* memory exhausted, try next interval */
        }

        memcpy(p->payload, tx_buf, len);
        err_t err = udp_send(s_stream_pcb, p);
        if (err == ERR_OK) {
            s_stats.stream_tx_count++;
            s_stats.stream_tx_bytes += len;
        }
        pbuf_free(p);
        sent++;
    }

    /* Re-arm timer — runs in tcpip_thread, no race condition */
    sys_timeout(UDP_STREAM_PERIOD_MS, stream_drain_timer, NULL);
}

/* ════════════════════════════════════════════════════════════════
 * Command Receive Callback (Port 51551 RX/TX)
 *
 * Called directly by LwIP when a UDP packet arrives on port 51551.
 * Executes in tcpip_thread — ZERO task-switch overhead.
 * This is the key latency advantage over the Netconn version.
 * ════════════════════════════════════════════════════════════════ */

static void cmd_recv_callback(void *arg, struct udp_pcb *pcb,
                               struct pbuf *p, const ip_addr_t *addr,
                               u16_t port)
{
    (void)arg;

    if (p == NULL) {
        return;
    }

    s_stats.cmd_rx_count++;

    /* Validate packet size */
    if (p->tot_len < 4 || p->tot_len > sizeof(udp_raw_cmd_packet_t)) {
        pbuf_free(p);
        return;
    }

    /* Copy request from pbuf (may be chained) */
    static udp_raw_cmd_packet_t req;
    static udp_raw_cmd_packet_t resp;

    memset(&req, 0, sizeof(req));
    pbuf_copy_partial(p, &req, p->tot_len, 0);
    pbuf_free(p);

    /* Invoke handler */
    memset(&resp, 0, sizeof(resp));
    if (s_cmd_handler != NULL) {
        s_cmd_handler(&req, &resp);
    } else {
        resp.cmd_id = req.cmd_id;
        resp.status = 0xFE; /* no handler registered */
        resp.payload_len = 0;
    }

    /* Send response immediately */
    uint16_t resp_len = 4 + resp.payload_len;
    if (resp_len > sizeof(udp_raw_cmd_packet_t)) {
        resp_len = sizeof(udp_raw_cmd_packet_t);
    }

    struct pbuf *resp_pbuf = pbuf_alloc(PBUF_TRANSPORT, resp_len, PBUF_RAM);
    if (resp_pbuf != NULL) {
        memcpy(resp_pbuf->payload, &resp, resp_len);
        udp_sendto(pcb, resp_pbuf, addr, port);
        s_stats.cmd_tx_count++;
        pbuf_free(resp_pbuf);
    }
}

/* ════════════════════════════════════════════════════════════════
 * Public API: Initialization
 * ════════════════════════════════════════════════════════════════ */

void udp_raw_cmd_register_handler(udp_raw_cmd_handler_t handler)
{
    s_cmd_handler = handler;
}

void udp_raw_server_init(void)
{
    /* ── Stream PCB (Port 51550 TX) ── */
    s_stream_pcb = udp_new();
    if (s_stream_pcb == NULL) {
        return; /* fatal: cannot allocate */
    }

    ip_addr_t remote_ip;
    ipaddr_aton(UDP_REMOTE_IP, &remote_ip);
    udp_connect(s_stream_pcb, &remote_ip, UDP_REMOTE_PORT);

    /* Start drain timer */
    sys_timeout(UDP_STREAM_PERIOD_MS, stream_drain_timer, NULL);

    /* ── Command PCB (Port 51551 RX/TX) ── */
    s_cmd_pcb = udp_new();
    if (s_cmd_pcb == NULL) {
        return; /* fatal: cannot allocate */
    }

    udp_bind(s_cmd_pcb, IP_ADDR_ANY, UDP_CMD_PORT);
    udp_recv(s_cmd_pcb, cmd_recv_callback, NULL);
}

void udp_raw_server_init_cb(void *arg)
{
    (void)arg;
    udp_raw_server_init();
}

/* ════════════════════════════════════════════════════════════════
 * Public API: Statistics
 * ════════════════════════════════════════════════════════════════ */

void udp_raw_get_stats(udp_raw_stats_t *out)
{
    if (out != NULL) {
        uint32_t primask = __get_PRIMASK();
        __disable_irq();
        *out = s_stats;
        if (!primask) __enable_irq();
    }
}

/* ════════════════════════════════════════════════════════════════
 * Example main() — Raw API version (udp_raw_server)
 *
 * Two FreeRTOS threads:
 *   Thread 1: control_task  — 1kHz control loop, streams data via udp_raw_print/udp_raw_printf
 *   Thread 2: monitor_task  — low-priority status logging
 *
 * Key difference from Netconn version:
 *   - UDP server does NOT create its own tasks (zero task overhead)
 *   - Initialization must happen in tcpip_thread via tcpip_callback()
 *   - Command-response runs in tcpip_thread recv callback (lowest latency)
 *   - Stream drain runs in tcpip_thread sys_timeout timer
 *
 * Build: include this file + udp_raw_server.c + eth_config.h
 * ════════════════════════════════════════════════════════════════ */
#if 0

#include "main.h"
#include "cmsis_os2.h"
#include "lwip/init.h"
#include "lwip/tcpip.h"
#include "lwip.h"
#include "udp_raw_server.h"
#include "eth_config.h"
#include <math.h>

/* ── Command handler (called in tcpip_thread context — NOT a user task) ── */
static void my_cmd_handler(const udp_raw_cmd_packet_t *req, udp_raw_cmd_packet_t *resp)
{
    resp->cmd_id = req->cmd_id;
    resp->status = 0; /* OK */

    switch (req->cmd_id) {
    case 0x01: { /* PING — echo back payload */
        resp->payload_len = req->payload_len;
        memcpy(resp->payload, req->payload, req->payload_len);
        break;
    }
    case 0x02: { /* GET_STATUS — return stats */
        udp_raw_stats_t stats;
        udp_raw_get_stats(&stats);
        resp->payload_len = sizeof(stats);
        memcpy(resp->payload, &stats, sizeof(stats));
        break;
    }
    default:
        resp->status = 0xFF; /* unknown command */
        resp->payload_len = 0;
        break;
    }
}

/* ── Thread 1: Control Task (1kHz, high priority) ── */
static void control_task(void *arg)
{
    (void)arg;
    uint32_t tick = 0;

    for (;;) {
        float t = (float)tick * 0.001f; /* seconds */

        /* Simulate sensor data */
        float position = sinf(2.0f * 3.14159f * 1.0f * t);
        float velocity = cosf(2.0f * 3.14159f * 1.0f * t);
        float current  = 0.5f * sinf(2.0f * 3.14159f * 5.0f * t);

        /* Method 1: Send as packed binary struct */
        typedef struct __attribute__((packed)) {
            uint32_t tick;
            float    pos;
            float    vel;
            float    cur;
        } stream_pkt_t;

        stream_pkt_t pkt = {tick, position, velocity, current};
        UDP_RAW_PRINT_STRUCT(pkt);

        /* Method 2: Send as formatted string */
        /* udp_raw_printf("[%lu] p=%.3f v=%.3f i=%.3f\n", tick, position, velocity, current); */

        tick++;
        osDelay(CONTROL_PERIOD_MS);
    }
}

/* ── Thread 2: Monitor Task (1Hz, low priority) ── */
static void monitor_task(void *arg)
{
    (void)arg;

    for (;;) {
        udp_raw_stats_t stats;
        udp_raw_get_stats(&stats);

        udp_raw_printf("[MONITOR] tx=%lu bytes=%lu overflow=%lu cmd_rx=%lu\n",
                       stats.stream_tx_count,
                       stats.stream_tx_bytes,
                       stats.stream_overflow,
                       stats.cmd_rx_count);

        osDelay(1000);
    }
}

/* ── main() ── */
int main(void)
{
    HAL_Init();
    SystemClock_Config();

    /* Peripherals & LwIP init (generated by CubeMX) */
    MX_GPIO_Init();
    MX_ETH_Init();
    MX_LWIP_Init();

    /* Initialize RTOS kernel */
    osKernelInitialize();

    /*
     * Register command handler BEFORE init.
     * Init MUST run in tcpip_thread context — use tcpip_callback().
     *
     * Unlike Netconn version, udp_raw_server_init() does NOT create
     * any FreeRTOS tasks. Everything runs inside tcpip_thread via
     * callbacks and sys_timeout timers.
     */
    udp_raw_cmd_register_handler(my_cmd_handler);
    tcpip_callback(udp_raw_server_init_cb, NULL);

    /* Create application threads */
    const osThreadAttr_t ctrl_attr = {
        .name       = "control",
        .stack_size = CONTROL_TASK_STACK * 4,
        .priority   = CONTROL_TASK_PRIO,
    };
    osThreadNew(control_task, NULL, &ctrl_attr);

    const osThreadAttr_t mon_attr = {
        .name       = "monitor",
        .stack_size = 512 * 4,
        .priority   = osPriorityLow,
    };
    osThreadNew(monitor_task, NULL, &mon_attr);

    /* Start scheduler — never returns */
    osKernelStart();

    for (;;) {}
}

#endif /* Example main() */
