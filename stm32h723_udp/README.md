# STM32H723 High-Speed Ethernet UDP Example

STM32H723 + LwIP + FreeRTOS + Netconn API 기반 고속 UDP 통신 예제 코드.

## Port 구성

| Port | 방향 | 용도 | 우선순위 |
|------|------|------|---------|
| 51550 | TX only | 1kHz 스트리밍 (udp_print) | BelowNormal |
| 51551 | RX/TX | Command-Response | AboveNormal |

## 파일 구조

```
Core/
├── Inc/
│   ├── eth_config.h        — 네트워크, 버퍼, 태스크 설정 매크로
│   ├── udp_server.h        — UDP 모듈 API (udp_print, 커맨드 핸들러)
│   └── control_thread.h    — 제어 스레드 API, 파라미터 구조체
├── Src/
│   ├── udp_server.c        — UDP 핵심 구현 (ring buffer + 2 tasks)
│   └── control_thread.c    — 1kHz 제어 루프 예제 (TIM6 trigger)
```

## Architecture

```
TIM6 ISR (1kHz)
    │
    ▼ osThreadFlagsSet
┌─────────────────────────────────────────────┐
│ control_task (osPriorityHigh, ~0.5ms/loop)  │
│   ADC → PI Control → PWM                   │
│   udp_print(current_data)  ──┐              │
│   udp_print(voltage_data)  ──┼─► Ring Buffer│
│   udp_print(debug_data)   ──┘  (8KB SPSC)  │
└─────────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────┐
                    │ udp_stream_task (BelowNormal)│
                    │   drain ring buf → UDP TX    │
                    │   → Port 51550               │
                    └─────────────────────────────┘

PC ──► Port 51551 ──► udp_cmd_task (AboveNormal) ──► handler callback ──► Response
```

## Task 우선순위

| Task | Priority | 주기 | 역할 |
|------|----------|------|------|
| control_task | High | 1ms (TIM6) | 모터 제어, 센서, udp_print |
| udp_cmd_task | AboveNormal | event | 커맨드 수신/응답 (Port 51551) |
| tcpip_thread | Normal | — | LwIP 내부 (ARP, DHCP 등) |
| udp_stream_task | BelowNormal | 1ms poll | Ring buffer drain → TX (Port 51550) |

## CubeMX 설정 (필수)

### ETH
- Mode: RMII
- ETH global interrupt: Enable

### LwIP
- DHCP: Disable
- IP: 192.168.1.2 / 255.255.255.0 / GW 192.168.1.1
- `LWIP_UDP = 1`, `LWIP_NETCONN = 1`
- `PBUF_POOL_SIZE = 24`, `MEM_SIZE = 16384`
- `LWIP_SO_RCVTIMEO = 1`

### FreeRTOS
- CMSIS_V2 API
- TOTAL_HEAP_SIZE >= 32768
- Tick Rate: 1000 Hz

### TIM6
- Prescaler: (APB1_CLK / 1000000) - 1 → 1MHz counter
- Period: 999 → 1kHz interrupt
- NVIC Enable, Preemption Priority >= 5

### MPU (STM32H7 필수)
D2 SRAM (0x30000000, 256KB): Device, Not Cacheable, Shareable.
이 설정 없으면 ETH DMA가 동작하지 않음.

### Linker Script
```ld
.lwip_sec (NOLOAD) :
{
  . = ABSOLUTE(0x30040000);
  *(.RxDecripSection)
  *(.TxDecripSection)
  *(.RxArraySection)
} >RAM_D2
```

## main.c 초기화 순서

```c
/* USER CODE BEGIN 2 */
MX_LWIP_Init();                        // CubeMX 생성
udp_cmd_register_handler(my_handler);  // 커맨드 핸들러 등록
udp_server_init();                     // UDP task 생성
control_thread_init();                 // 제어 task + TIM6 시작
/* USER CODE END 2 */
```

## udp_print() 사용법

```c
// control loop 내에서 여러 번 호출 가능
udp_print(&current_pkt, sizeof(current_pkt));   // 전류 데이터
udp_print(&voltage_pkt, sizeof(voltage_pkt));   // 전압 데이터
if (fault) udp_print(&debug_pkt, sizeof(debug_pkt));  // 조건부
```

모든 호출은 ring buffer에 enqueue되며, stream task가 전부 송신.
Ring buffer overflow 시에만 데이터 드롭 (stats.stream_overflow로 확인).

## 커맨드 프로토콜 (Port 51551)

```
Request/Response format (packed):
┌──────────┬──────────┬──────────────┬─────────────────┐
│ cmd_id   │ status   │ payload_len  │ payload[504]    │
│ (1 byte) │ (1 byte) │ (2 bytes LE) │ (variable)      │
└──────────┴──────────┴──────────────┴─────────────────┘
```

| cmd_id | 설명 | Payload (req) | Payload (resp) |
|--------|------|---------------|----------------|
| 0x01 | Read params | - | control_params_t |
| 0x02 | Write params | kp(4B) + ki(4B) + i_ref(4B) | - |
| 0x03 | Read status | - | i_meas(4B) + v_out(4B) |

## PC 테스트 (Python)

```python
import socket, struct

# Port 51550 수신 (스트리밍)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 51550))
while True:
    data, addr = sock.recvfrom(256)
    pkt_id = data[0]
    if pkt_id == 0x01:  # current
        ts, i_ref, i_meas = struct.unpack_from('<Iff', data, 1)
        print(f"t={ts} i_ref={i_ref:.3f} i_meas={i_meas:.3f}")

# Port 51551 커맨드 (Read params)
cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
req = struct.pack('<BBH', 0x01, 0, 0)  # cmd_id=1, status=0, len=0
cmd_sock.sendto(req, ('192.168.1.2', 51551))
resp, _ = cmd_sock.recvfrom(512)
cmd_id, status, plen = struct.unpack_from('<BBH', resp, 0)
print(f"cmd={cmd_id} status={status} payload_len={plen}")
```
