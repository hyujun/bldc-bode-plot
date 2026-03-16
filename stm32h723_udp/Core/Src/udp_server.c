/**
 * @file udp_server.c
 * @brief High-speed UDP server implementation (LwIP Netconn + FreeRTOS)
 *
 * Port 51550: TX-only streaming via lock-free SPSC ring buffer
 * Port 51551: Low-latency command-response
 */

#include "udp_server.h"
#include "eth_config.h"

#include "cmsis_os2.h"
#include "lwip/api.h"
#include "lwip/netbuf.h"
#include "lwip/ip_addr.h"

#include <string.h>
#include <stdio.h>
#include <stdarg.h>

/* ════════════════════════════════════════════════════════════════
 * Lock-free SPSC Ring Buffer
 *
 * Single producer (control_task via udp_print) →
 * Single consumer (udp_stream_task).
 *
 * Each entry: [uint16_t length][payload bytes]
 * head/tail are only modified by their respective owner.
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
    /* keep 1 byte gap to distinguish full from empty */
    return UDP_RING_BUF_SIZE - 1 - ring_readable(rb);
}

/**
 * @brief Write [len(2B) | data(len B)] into ring buffer.
 * @return 0 on success, -1 on overflow (data dropped)
 */
static int ring_write(ring_buf_t *rb, const void *data, uint16_t len)
{
    uint32_t total = (uint32_t)len + 2u; /* 2-byte length prefix */
    if (ring_writable(rb) < total) {
        return -1; /* overflow */
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

    /* Memory barrier before publishing head */
    __DMB();
    rb->head = h;

    return 0;
}

/**
 * @brief Read one entry from ring buffer.
 * @param out      Destination buffer
 * @param max_len  Maximum bytes to read
 * @return Actual payload length, 0 if empty
 */
static uint16_t ring_read(ring_buf_t *rb, void *out, uint16_t max_len)
{
    if (ring_readable(rb) < 2) {
        return 0; /* empty */
    }

    uint32_t t = rb->tail;

    /* Read length prefix */
    uint16_t len = rb->buf[t];
    t = (t + 1) % UDP_RING_BUF_SIZE;
    len |= (uint16_t)rb->buf[t] << 8;
    t = (t + 1) % UDP_RING_BUF_SIZE;

    if (len == 0 || ring_readable(rb) < (uint32_t)(len + 2)) {
        /* Corrupted or incomplete — should not happen with correct producer */
        return 0;
    }

    /* Read payload */
    uint8_t *dst = (uint8_t *)out;
    uint16_t copy_len = (len <= max_len) ? len : max_len;

    for (uint16_t i = 0; i < len; i++) {
        if (i < copy_len) {
            dst[i] = rb->buf[t];
        }
        t = (t + 1) % UDP_RING_BUF_SIZE;
    }

    /* Memory barrier before publishing tail */
    __DMB();
    rb->tail = t;

    return copy_len;
}

/* ════════════════════════════════════════════════════════════════
 * Static State
 * ════════════════════════════════════════════════════════════════ */

static udp_cmd_handler_t s_cmd_handler = NULL;
static udp_stats_t       s_stats       = {0};

/* ════════════════════════════════════════════════════════════════
 * Public API: udp_print()
 * ════════════════════════════════════════════════════════════════ */

void udp_print(const void *data, uint16_t len)
{
    if (data == NULL || len == 0 || len > UDP_STREAM_MAX_PKT) {
        return;
    }

    if (ring_write(&s_ring, data, len) != 0) {
        s_stats.stream_overflow++;
    }
}

/* ════════════════════════════════════════════════════════════════
 * Public API: udp_printf()
 * ════════════════════════════════════════════════════════════════ */

int udp_printf(const char *fmt, ...)
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

    /* Truncate to buffer size (vsnprintf returns what it *would* have written) */
    uint16_t send_len = (len < (int)sizeof(buf)) ? (uint16_t)len : (uint16_t)(sizeof(buf) - 1);

    udp_print(buf, send_len);
    return (int)send_len;
}

/* ════════════════════════════════════════════════════════════════
 * Stream Task (Port 51550 TX)
 *
 * Drains the ring buffer and sends each entry as an independent
 * UDP packet. Runs at BelowNormal priority — yields to control
 * and command tasks.
 * ════════════════════════════════════════════════════════════════ */

static void udp_stream_task(void *arg)
{
    (void)arg;

    /* Create netconn and connect to remote */
    struct netconn *conn = netconn_new(NETCONN_UDP);
    if (conn == NULL) {
        for (;;) { osDelay(1000); } /* fatal: cannot allocate */
    }

    ip_addr_t remote_ip;
    ipaddr_aton(UDP_REMOTE_IP, &remote_ip);
    netconn_connect(conn, &remote_ip, UDP_REMOTE_PORT);

    static uint8_t tx_buf[UDP_STREAM_MAX_PKT];

    for (;;) {
        uint32_t sent = 0;

        /* Drain up to DRAIN_MAX packets per loop */
        while (sent < UDP_STREAM_DRAIN_MAX) {
            uint16_t len = ring_read(&s_ring, tx_buf, sizeof(tx_buf));
            if (len == 0) {
                break; /* ring empty */
            }

            /* Allocate netbuf and send */
            struct netbuf *nbuf = netbuf_new();
            if (nbuf == NULL) {
                break; /* memory exhausted, try next loop */
            }

            void *payload = netbuf_alloc(nbuf, len);
            if (payload != NULL) {
                memcpy(payload, tx_buf, len);
                err_t err = netconn_send(conn, nbuf);
                if (err == ERR_OK) {
                    s_stats.stream_tx_count++;
                    s_stats.stream_tx_bytes += len;
                }
            }
            netbuf_delete(nbuf);
            sent++;
        }

        /* Yield CPU when ring is empty */
        if (sent == 0) {
            osDelay(UDP_STREAM_PERIOD_MS);
        }
    }
}

/* ════════════════════════════════════════════════════════════════
 * Command Task (Port 51551 RX/TX)
 *
 * Listens for incoming commands, invokes the registered handler,
 * and sends the response back to the requester.
 * Runs at AboveNormal priority for low latency.
 * ════════════════════════════════════════════════════════════════ */

static void udp_cmd_task(void *arg)
{
    (void)arg;

    /* Create netconn, bind to command port */
    struct netconn *conn = netconn_new(NETCONN_UDP);
    if (conn == NULL) {
        for (;;) { osDelay(1000); }
    }

    netconn_bind(conn, IP_ADDR_ANY, UDP_CMD_PORT);
    netconn_set_recvtimeout(conn, UDP_CMD_RECV_TIMEOUT);

    static udp_cmd_packet_t req;
    static udp_cmd_packet_t resp;

    for (;;) {
        struct netbuf *nbuf = NULL;
        err_t err = netconn_recv(conn, &nbuf);

        if (err != ERR_OK) {
            /* Timeout or error — just retry */
            if (nbuf) netbuf_delete(nbuf);
            continue;
        }

        s_stats.cmd_rx_count++;

        /* Extract request data */
        void *data_ptr = NULL;
        u16_t data_len = 0;
        netbuf_data(nbuf, &data_ptr, &data_len);

        if (data_ptr == NULL || data_len < 4 || data_len > sizeof(udp_cmd_packet_t)) {
            netbuf_delete(nbuf);
            continue;
        }

        /* Copy into local struct */
        memset(&req, 0, sizeof(req));
        memcpy(&req, data_ptr, data_len);

        /* Get sender address for response */
        const ip_addr_t *remote_addr = netbuf_fromaddr(nbuf);
        u16_t remote_port = netbuf_fromport(nbuf);

        netbuf_delete(nbuf);
        nbuf = NULL;

        /* Invoke handler */
        memset(&resp, 0, sizeof(resp));
        if (s_cmd_handler != NULL) {
            s_cmd_handler(&req, &resp);
        } else {
            resp.cmd_id = req.cmd_id;
            resp.status = 0xFE; /* no handler registered */
            resp.payload_len = 0;
        }

        /* Send response */
        uint16_t resp_len = 4 + resp.payload_len; /* header(4) + payload */
        if (resp_len > sizeof(udp_cmd_packet_t)) {
            resp_len = sizeof(udp_cmd_packet_t);
        }

        struct netbuf *resp_buf = netbuf_new();
        if (resp_buf != NULL) {
            void *resp_payload = netbuf_alloc(resp_buf, resp_len);
            if (resp_payload != NULL) {
                memcpy(resp_payload, &resp, resp_len);
                netconn_sendto(conn, resp_buf, remote_addr, remote_port);
                s_stats.cmd_tx_count++;
            }
            netbuf_delete(resp_buf);
        }
    }
}

/* ════════════════════════════════════════════════════════════════
 * Public API
 * ════════════════════════════════════════════════════════════════ */

void udp_cmd_register_handler(udp_cmd_handler_t handler)
{
    s_cmd_handler = handler;
}

void udp_server_init(void)
{
    /* Stream task — low priority, drains ring buffer */
    const osThreadAttr_t stream_attr = {
        .name       = "udp_stream",
        .stack_size = UDP_STREAM_TASK_STACK * 4,
        .priority   = UDP_STREAM_TASK_PRIO,
    };
    osThreadNew(udp_stream_task, NULL, &stream_attr);

    /* Command task — high priority, fast response */
    const osThreadAttr_t cmd_attr = {
        .name       = "udp_cmd",
        .stack_size = UDP_CMD_TASK_STACK * 4,
        .priority   = UDP_CMD_TASK_PRIO,
    };
    osThreadNew(udp_cmd_task, NULL, &cmd_attr);
}

void udp_get_stats(udp_stats_t *out)
{
    if (out != NULL) {
        /* Disable interrupts briefly for consistent snapshot */
        uint32_t primask = __get_PRIMASK();
        __disable_irq();
        *out = s_stats;
        if (!primask) __enable_irq();
    }
}

/* ════════════════════════════════════════════════════════════════
 * Example main() — Netconn API version (udp_server)
 *
 * Two FreeRTOS threads:
 *   Thread 1: control_task  — 1kHz control loop, streams data via udp_print/udp_printf
 *   Thread 2: monitor_task  — low-priority status logging
 *
 * Build: include this file + udp_server.c + eth_config.h
 * ════════════════════════════════════════════════════════════════ */
#if 0

#include "main.h"
#include "cmsis_os2.h"
#include "lwip/init.h"
#include "lwip.h"
#include "udp_server.h"
#include "eth_config.h"
#include <math.h>

/* ── Command handler (called in udp_cmd_task context) ── */
static void my_cmd_handler(const udp_cmd_packet_t *req, udp_cmd_packet_t *resp)
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
        udp_stats_t stats;
        udp_get_stats(&stats);
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
        UDP_PRINT_STRUCT(pkt);

        /* Method 2: Send as formatted string */
        /* udp_printf("[%lu] p=%.3f v=%.3f i=%.3f\n", tick, position, velocity, current); */

        tick++;
        osDelay(CONTROL_PERIOD_MS);
    }
}

/* ── Thread 2: Monitor Task (1Hz, low priority) ── */
static void monitor_task(void *arg)
{
    (void)arg;

    for (;;) {
        udp_stats_t stats;
        udp_get_stats(&stats);

        udp_printf("[MONITOR] tx=%lu bytes=%lu overflow=%lu cmd_rx=%lu\n",
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

    /* Register command handler & start UDP server (creates stream + cmd tasks internally) */
    udp_cmd_register_handler(my_cmd_handler);
    udp_server_init();

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
