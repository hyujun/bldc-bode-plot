/**
 * @file eth_config.h
 * @brief Ethernet UDP configuration macros for STM32H723
 *
 * Network, buffer, task priority, and timing parameters.
 */

#ifndef ETH_CONFIG_H
#define ETH_CONFIG_H

/* ════════════════════════════════════════════════════════════════
 * Network Configuration
 * ════════════════════════════════════════════════════════════════ */
#define UDP_STREAM_PORT       51550         /* TX-only streaming port */
#define UDP_CMD_PORT          51551         /* Command-Response port */
#define UDP_REMOTE_IP         "192.168.1.1" /* PC IP (destination for streaming) */
#define UDP_REMOTE_PORT       51550         /* PC receive port */

/* ════════════════════════════════════════════════════════════════
 * Ring Buffer (Port 51550 Streaming)
 * ════════════════════════════════════════════════════════════════ */
#define UDP_RING_BUF_SIZE     8192  /* 8KB SPSC ring buffer */
#define UDP_STREAM_MAX_PKT    256   /* Max bytes per udp_print() call */
#define UDP_STREAM_DRAIN_MAX  16    /* Max packets drained per stream task loop */

/* ════════════════════════════════════════════════════════════════
 * Command Buffer (Port 51551)
 * ════════════════════════════════════════════════════════════════ */
#define UDP_CMD_BUF_SIZE      512   /* Max command/response packet size (bytes) */

/* ════════════════════════════════════════════════════════════════
 * FreeRTOS Task Configuration
 * ════════════════════════════════════════════════════════════════ */

/* Priorities (higher = more urgent)
 * control_task:  osPriorityHigh        — must never be preempted by UDP
 * cmd_task:      osPriorityAboveNormal — fast request-response
 * stream_task:   osPriorityBelowNormal — best-effort streaming
 */
#define CONTROL_TASK_PRIO     (osPriorityHigh)
#define UDP_CMD_TASK_PRIO     (osPriorityAboveNormal)
#define UDP_STREAM_TASK_PRIO  (osPriorityBelowNormal)

/* Stack sizes in words (1 word = 4 bytes) */
#define CONTROL_TASK_STACK    512   /* 2KB */
#define UDP_CMD_TASK_STACK    512   /* 2KB */
#define UDP_STREAM_TASK_STACK 512   /* 2KB */

/* ════════════════════════════════════════════════════════════════
 * Timing
 * ════════════════════════════════════════════════════════════════ */
#define CONTROL_PERIOD_MS     1     /* 1kHz control loop (TIM6 driven) */
#define UDP_STREAM_PERIOD_MS  1     /* Stream task poll interval */
#define UDP_CMD_RECV_TIMEOUT  10    /* ms, netconn recv timeout */

#endif /* ETH_CONFIG_H */
