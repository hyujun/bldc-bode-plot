/**
 * @file udp_server.h
 * @brief High-speed UDP server for STM32H723 (LwIP + FreeRTOS + Netconn)
 *
 * Two UDP ports:
 *   Port 51550 — TX-only streaming via udp_print() + ring buffer
 *   Port 51551 — Low-latency command-response
 */

#ifndef UDP_SERVER_H
#define UDP_SERVER_H

#include <stdint.h>
#include <stddef.h>

/* ════════════════════════════════════════════════════════════════
 * Streaming API (Port 51550)
 * ════════════════════════════════════════════════════════════════ */

/**
 * @brief Enqueue arbitrary binary data for UDP streaming.
 *
 * Can be called multiple times per control loop iteration.
 * Each call produces one independent UDP packet on port 51550.
 * Data is buffered in a lock-free SPSC ring buffer and drained
 * by the stream task — no data is lost unless the ring overflows.
 *
 * @param data  Pointer to data to send
 * @param len   Length in bytes (max UDP_STREAM_MAX_PKT)
 */
void udp_print(const void *data, uint16_t len);

/** Convenience macro: send a packed struct */
#define UDP_PRINT_STRUCT(s)  udp_print(&(s), sizeof(s))

/* ════════════════════════════════════════════════════════════════
 * Command-Response API (Port 51551)
 * ════════════════════════════════════════════════════════════════ */

/** Command/response packet (packed, max 508 bytes total) */
typedef struct __attribute__((packed)) {
    uint8_t  cmd_id;           /**< Command identifier */
    uint8_t  status;           /**< Response status (0 = OK) */
    uint16_t payload_len;      /**< Payload length in bytes */
    uint8_t  payload[504];     /**< Variable-length payload */
} udp_cmd_packet_t;

/**
 * @brief Command handler callback type.
 *
 * Called in the cmd_task context when a request arrives on port 51551.
 * The handler must populate resp (cmd_id, status, payload, payload_len).
 * The response is sent back to the requester immediately after return.
 *
 * @param req   Received command packet (read-only)
 * @param resp  Response packet to fill (pre-zeroed)
 */
typedef void (*udp_cmd_handler_t)(const udp_cmd_packet_t *req,
                                  udp_cmd_packet_t *resp);

/**
 * @brief Register the command handler callback.
 *
 * Must be called before udp_server_init(). Only one handler is supported.
 *
 * @param handler  Function pointer to the command handler
 */
void udp_cmd_register_handler(udp_cmd_handler_t handler);

/* ════════════════════════════════════════════════════════════════
 * Initialization & Statistics
 * ════════════════════════════════════════════════════════════════ */

/**
 * @brief Initialize the UDP server.
 *
 * Creates two FreeRTOS tasks:
 *   - udp_stream_task (BelowNormal priority) — drains ring buffer → port 51550
 *   - udp_cmd_task (AboveNormal priority) — listens on port 51551
 *
 * Call after MX_LWIP_Init() and udp_cmd_register_handler().
 */
void udp_server_init(void);

/** Runtime statistics (read-only snapshot) */
typedef struct {
    uint32_t stream_tx_count;     /**< Packets sent on port 51550 */
    uint32_t stream_tx_bytes;     /**< Total bytes sent on port 51550 */
    uint32_t stream_overflow;     /**< Ring buffer overflow count (dropped) */
    uint32_t cmd_rx_count;        /**< Commands received on port 51551 */
    uint32_t cmd_tx_count;        /**< Responses sent on port 51551 */
} udp_stats_t;

/**
 * @brief Get a snapshot of runtime statistics.
 * @param out  Destination struct (copied atomically)
 */
void udp_get_stats(udp_stats_t *out);

#endif /* UDP_SERVER_H */
