/**
 * @file udp_raw_server.h
 * @brief High-speed UDP server using LwIP Raw API (callback-based, minimal latency)
 *
 * Drop-in alternative to udp_server.h (Netconn version).
 * No FreeRTOS tasks required — runs entirely in tcpip_thread context.
 *
 * Two UDP ports:
 *   Port 51550 — TX-only streaming via udp_raw_print() + ring buffer
 *   Port 51551 — Ultra-low-latency command-response (recv callback)
 */

#ifndef UDP_RAW_SERVER_H
#define UDP_RAW_SERVER_H

#include <stdint.h>
#include <stddef.h>
#include <stdarg.h>

/* ════════════════════════════════════════════════════════════════
 * Streaming API (Port 51550)
 * ════════════════════════════════════════════════════════════════ */

/**
 * @brief Enqueue arbitrary binary data for UDP streaming.
 *
 * Thread-safe: can be called from any context (ISR, task, main loop).
 * Data is buffered in a lock-free SPSC ring buffer and drained
 * periodically via software timer in tcpip_thread context.
 *
 * @param data  Pointer to data to send
 * @param len   Length in bytes (max UDP_STREAM_MAX_PKT)
 */
void udp_raw_print(const void *data, uint16_t len);

/** Convenience macro: send a packed struct */
#define UDP_RAW_PRINT_STRUCT(s)  udp_raw_print(&(s), sizeof(s))

/**
 * @brief printf-style formatted string output over UDP (port 51550).
 *
 * Formats the string into an internal buffer and enqueues it via udp_raw_print().
 * Max output length is UDP_STREAM_MAX_PKT bytes (truncated if exceeded).
 *
 * @param fmt  printf format string
 * @param ...  format arguments
 * @return     Number of characters sent (excluding null terminator), or -1 on error
 */
int udp_raw_printf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

/* ════════════════════════════════════════════════════════════════
 * Command-Response API (Port 51551)
 * ════════════════════════════════════════════════════════════════ */

/** Command/response packet (packed, max 508 bytes total) */
typedef struct __attribute__((packed)) {
    uint8_t  cmd_id;           /**< Command identifier */
    uint8_t  status;           /**< Response status (0 = OK) */
    uint16_t payload_len;      /**< Payload length in bytes */
    uint8_t  payload[504];     /**< Variable-length payload */
} udp_raw_cmd_packet_t;

/**
 * @brief Command handler callback type.
 *
 * Called directly in the tcpip_thread recv callback — executes with
 * ZERO task-switch overhead for minimum latency.
 *
 * IMPORTANT: Handler must return quickly. Do NOT call blocking APIs.
 *
 * @param req   Received command packet (read-only)
 * @param resp  Response packet to fill (pre-zeroed)
 */
typedef void (*udp_raw_cmd_handler_t)(const udp_raw_cmd_packet_t *req,
                                      udp_raw_cmd_packet_t *resp);

/**
 * @brief Register the command handler callback.
 *
 * Must be called before udp_raw_server_init(). Only one handler is supported.
 *
 * @param handler  Function pointer to the command handler
 */
void udp_raw_cmd_register_handler(udp_raw_cmd_handler_t handler);

/* ════════════════════════════════════════════════════════════════
 * Initialization & Statistics
 * ════════════════════════════════════════════════════════════════ */

/**
 * @brief Initialize the Raw API UDP server.
 *
 * Creates two UDP PCBs:
 *   - Stream PCB — connected to remote, drained by sys_timeout timer
 *   - Command PCB — bound to port 51551, recv callback registered
 *
 * Must be called from tcpip_thread context (e.g., inside
 * tcpip_callback() or after netif is up in the tcpip init done callback).
 *
 * If using FreeRTOS + LwIP (NO_SYS=0), wrap the call:
 *   tcpip_callback(udp_raw_server_init_cb, NULL);
 */
void udp_raw_server_init(void);

/**
 * @brief Wrapper for tcpip_callback()-safe initialization.
 * @param arg  Unused (pass NULL)
 */
void udp_raw_server_init_cb(void *arg);

/** Runtime statistics (read-only snapshot) */
typedef struct {
    uint32_t stream_tx_count;     /**< Packets sent on port 51550 */
    uint32_t stream_tx_bytes;     /**< Total bytes sent on port 51550 */
    uint32_t stream_overflow;     /**< Ring buffer overflow count (dropped) */
    uint32_t cmd_rx_count;        /**< Commands received on port 51551 */
    uint32_t cmd_tx_count;        /**< Responses sent on port 51551 */
} udp_raw_stats_t;

/**
 * @brief Get a snapshot of runtime statistics.
 * @param out  Destination struct (copied atomically)
 */
void udp_raw_get_stats(udp_raw_stats_t *out);

#endif /* UDP_RAW_SERVER_H */
