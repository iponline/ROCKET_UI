import serial

def read_telemetry_packet(ser):
    while True:
        b = ser.read(1)
        if not b:
            print("I'm here 2 (timeout)")
            continue

        val = b[0]
        if val != 0xAA:
            # show what you’re actually getting
            print(f"Desync byte: 0x{val:02X}")
            continue

        print("I'm here 3 (found start 0xAA)")
        header = ser.read(2)
        if len(header) < 2:
            print("Short header")
            continue
        pkt_type, payload_len = header[0], header[1]

        data = ser.read(payload_len + 1)  # +1 for checksum (ignored)
        if len(data) < payload_len + 1:
            print("Short payload")
            continue
        payload = data[:-1]

        print(f"Received packet: TYPE=0x{pkt_type:02X} LEN={payload_len} PAYLOAD={payload.hex().upper()}")

if __name__ == '__main__':
    with serial.Serial('COM4', 115200, timeout=1) as ser:
        ser.reset_input_buffer()
        print("Listening for telemetry packets...")
        read_telemetry_packet(ser)
