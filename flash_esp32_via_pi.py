"""flash_esp32_via_pi.py - Wirelessly update the enclosed ESP32 over the 3-wire Pi UART.

No USB cable needed once housed! Connects to the Raspberry Pi over Wi-Fi/Ethernet,
temporarily pauses asl-device.service so /dev/serial0 is free, streams CRC32-verified
chunks to esp32_firmware.py's OTA handler, reboots the ESP32, verifies INFO, and
restarts asl-device.service.

Usage:
    .venv\\Scripts\\python.exe flash_esp32_via_pi.py           # flash main.py
    .venv\\Scripts\\python.exe flash_esp32_via_pi.py --font    # flash ar_font.bin
    .venv\\Scripts\\python.exe flash_esp32_via_pi.py --all     # flash both
"""
import argparse
import os
import shlex
import sys
import time

import paramiko

USER, PASS = "pi", "raspberry"
HOSTS = ["raspberrypi.local", "169.254.83.50", "192.168.0.23"]
HERE = os.path.dirname(os.path.abspath(__file__))

PI_SENDER = r'''#!/usr/bin/env python3
import base64, json, sys, time, zlib
import serial

PORT = "/dev/serial0"
BAUD = 115200
CHUNK_SIZE = 135  # multiple of 3 -> 180 base64 chars, zero padding overhead mid-stream

def read_line(ser, timeout=2.0):
    t0 = time.time()
    buf = bytearray()
    while time.time() - t0 < timeout:
        b = ser.read(1)
        if not b:
            continue
        if b in (b"\n", b"\r"):
            if buf:
                return buf.decode("utf-8", "ignore").strip()
        else:
            buf.extend(b)
    return ""

def wait_for(ser, prefix, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        ln = read_line(ser, timeout=max(0.1, timeout - (time.time() - t0)))
        if not ln:
            continue
        if ln.startswith(prefix):
            return ln
        if ln.startswith("OTA:ERR") or ln.startswith("ERR:"):
            raise RuntimeError(f"ESP32 error: {ln}")
    return ""

def send_cmd(ser, line):
    ser.write((line + "\n").encode("ascii"))
    ser.flush()

def main():
    manifest = json.loads(sys.argv[1])
    ser = serial.Serial(PORT, BAUD, timeout=0.05)
    try:
        ser.reset_input_buffer()
        pong = ""
        for _ in range(5):
            send_cmd(ser, "PING")
            pong = wait_for(ser, "PONG", timeout=0.8)
            if pong:
                break
        if not pong:
            raise RuntimeError("No PONG from ESP32 over /dev/serial0 (check 3 wires / firmware v11+)")
        send_cmd(ser, "INFO")
        info_before = wait_for(ser, "INFO:", timeout=1.5)
        print("[OTA] Connected:", info_before or "PONG OK", flush=True)

        for local_path, dest_name in manifest:
            data = open(local_path, "rb").read()
            crc = zlib.crc32(data) & 0xFFFFFFFF
            print(f"[OTA] Sending {dest_name} ({len(data)} bytes, CRC32={crc:08X})...", flush=True)
            send_cmd(ser, f"OTA:START:{dest_name}:{len(data)}")
            if not wait_for(ser, "OTA:READY", timeout=3.0):
                raise RuntimeError("ESP32 did not acknowledge OTA:START")

            total_chunks = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE
            for seq in range(total_chunks):
                chunk = data[seq * CHUNK_SIZE:(seq + 1) * CHUNK_SIZE]
                b64 = base64.b64encode(chunk).decode("ascii")
                ack_ok = False
                for _ in range(3):
                    send_cmd(ser, f"OTA:DATA:{seq}:{b64}")
                    ack = wait_for(ser, f"OTA:ACK:{seq}", timeout=2.0)
                    if ack == f"OTA:ACK:{seq}":
                        ack_ok = True
                        break
                if not ack_ok:
                    send_cmd(ser, "OTA:ABORT")
                    raise RuntimeError(f"Chunk {seq}/{total_chunks} failed ACK")
                if (seq + 1) % 40 == 0 or seq + 1 == total_chunks:
                    pct = int((seq + 1) * 100 / total_chunks)
                    print(f"  progress: {seq + 1}/{total_chunks} ({pct}%)", flush=True)

            send_cmd(ser, f"OTA:END:{crc:08X}")
            ok = wait_for(ser, "OTA:OK", timeout=4.0)
            if not ok:
                raise RuntimeError("ESP32 did not confirm OTA:OK")
            print(f"[OTA] Verified on flash: {dest_name}", flush=True)

        print("[OTA] Rebooting ESP32...", flush=True)
        send_cmd(ser, "OTA:REBOOT")
        wait_for(ser, "OTA:BYE", timeout=1.5)
        ready = wait_for(ser, "READY", timeout=8.0)
        if not ready:
            print("[OTA] No READY yet (the screen may be showing touch calibration - tap the crosses).", flush=True)
            return
        send_cmd(ser, "INFO")
        info_after = wait_for(ser, "INFO:", timeout=2.0)
        print("[OTA] Post-reboot:", info_after or ready, flush=True)
    finally:
        ser.close()

if __name__ == "__main__":
    main()
'''


def connect(explicit_host):
    cands = ([explicit_host] if explicit_host else []) + HOSTS
    for h in cands:
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(h, username=USER, password=PASS, timeout=6)
            print(f"[OTA] SSH connected to {h}", flush=True)
            return ssh
        except Exception:
            continue
    sys.exit("[OTA] ERROR: cannot reach Raspberry Pi via SSH")


def ssh_run(ssh, cmd, sudo=False):
    full = f"sudo -S -p '' bash -c {shlex.quote(cmd)}" if sudo else cmd
    stdin, stdout, stderr = ssh.exec_command(full)
    if sudo:
        stdin.write(PASS + "\n")
        stdin.flush()
    out = stdout.read().decode(errors="replace") + stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    return code, out


def main():
    ap = argparse.ArgumentParser(description="Update ESP32 firmware/font wirelessly over Raspberry Pi 3-wire UART")
    ap.add_argument("--pi", default=None, help="Pi hostname or IP")
    ap.add_argument("--font", action="store_true", help="Flash ar_font.bin only")
    ap.add_argument("--all", action="store_true", help="Flash both esp32_firmware.py (main.py) and ar_font.bin")
    args = ap.parse_args()

    files = []
    if args.all or not args.font:
        files.append(("esp32_firmware.py", "main.py"))
    if args.all or args.font:
        files.append(("ar_font.bin", "ar_font.bin"))

    for src, _ in files:
        if not os.path.exists(os.path.join(HERE, src)):
            sys.exit(f"[OTA] Missing local file: {src}")

    ssh = connect(args.pi)
    try:
        ssh_run(ssh, "mkdir -p /tmp/esp_ota")
        sftp = ssh.open_sftp()
        manifest = []
        for src, dest in files:
            rpath = f"/tmp/esp_ota/{dest}"
            sftp.put(os.path.join(HERE, src), rpath)
            manifest.append([rpath, dest])
        with sftp.open("/tmp/esp_ota/ota_send.py", "w") as f:
            f.write(PI_SENDER)
        sftp.close()

        print("[OTA] Pausing asl-device.service to free /dev/serial0...", flush=True)
        ssh_run(ssh, "systemctl stop asl-device", sudo=True)
        time.sleep(0.6)
        import json as _json
        cmd = f"python3 -u /tmp/esp_ota/ota_send.py {shlex.quote(_json.dumps(manifest))}"
        code, out = ssh_run(ssh, cmd)
        if out.strip():
            print(out.rstrip(), flush=True)
        if code != 0:
            sys.exit(f"[OTA] FAILED (exit {code})")
    finally:
        print("[OTA] Restarting asl-device.service...", flush=True)
        ssh_run(ssh, "rm -rf /tmp/esp_ota && systemctl start asl-device", sudo=True)
        time.sleep(3)
        _, status_out = ssh_run(ssh, "systemctl is-active asl-device; journalctl -u asl-device -n 8 --no-pager -o cat")
        print(status_out.rstrip(), flush=True)
        ssh.close()


if __name__ == "__main__":
    main()
