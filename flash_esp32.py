"""flash_esp32.py - upload esp32_firmware.py to the CYD as main.py.

Usage:  .venv\\Scripts\\python.exe flash_esp32.py [COM15]

The firmware disables Ctrl-C, so plain `mpremote` cannot interrupt it.
1. Ask the running firmware to exit to the REPL ("REPL" command), then copy.
2. If that fails (old firmware), wipe the filesystem header with esptool,
   which makes MicroPython boot to a bare REPL, then copy.
"""
import subprocess, sys, time
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM15"
PY = sys.executable

def mpremote(*args):
    return subprocess.run([PY, "-m", "mpremote", "connect", PORT, *args],
                          capture_output=True, text=True, timeout=60)

def ask_repl():
    s = serial.Serial(); s.port = PORT; s.baudrate = 115200
    s.dtr = False; s.rts = False; s.timeout = 0.2
    s.open(); time.sleep(0.3)
    s.write(b"\nREPL\n"); s.flush(); time.sleep(0.5)
    s.close()

def copy():
    r = mpremote("cp", "esp32_firmware.py", ":main.py")
    return r.returncode == 0, r.stdout + r.stderr

print(f"[flash] Asking firmware on {PORT} to release the REPL...")
ask_repl()
ok, log = copy()
if not ok:
    print("[flash] Firmware did not release; wiping filesystem header with esptool...")
    subprocess.run([PY, "-m", "esptool", "--port", PORT, "erase-region", "0x200000", "0x4000"], check=True)
    time.sleep(2.5)
    for attempt in range(3):
        ok, log = copy()
        if ok:
            break
        time.sleep(1.5)
if not ok:
    print(log); sys.exit("[flash] FAILED")
mpremote("reset")
print("[flash] Done. Device rebooting into new firmware.")
