"""deploy_pi.py - install / update the device program on the Raspberry Pi.

    .venv\\Scripts\\python.exe deploy_pi.py [pi-address]

Copies pi_device.py to /home/pi/asl, installs python3-serial, turns Wi-Fi power
saving off, and installs a systemd service so the device starts at every boot.
"""
import os
import shlex
import sys

import paramiko

USER, PASS = "pi", "raspberry"
HOSTS = [a for a in sys.argv[1:2]] + ["raspberrypi.local", "192.168.0.23", "169.254.83.50"]
HERE = os.path.dirname(os.path.abspath(__file__))

SERVICE = """[Unit]
Description=ASL Communicator device
After=network-online.target sound.target
Wants=network-online.target

[Service]
User=pi
WorkingDirectory=/home/pi/asl
ExecStart=/usr/bin/python3 -u /home/pi/asl/pi_device.py
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
"""


def connect():
    for h in HOSTS:
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(h, username=USER, password=PASS, timeout=6)
            print(f"[deploy] connected to {h}")
            return ssh
        except Exception:
            continue
    sys.exit("[deploy] cannot reach the Pi")


def run(ssh, cmd, sudo=False, show=True):
    full = f"sudo -S -p '' bash -c {shlex.quote(cmd)}" if sudo else cmd
    stdin, stdout, stderr = ssh.exec_command(full)
    if sudo:
        stdin.write(PASS + "\n"); stdin.flush()
    out = stdout.read().decode(errors="replace") + stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    if show and out.strip():
        print(out.rstrip())
    return code, out


def main():
    ssh = connect()
    run(ssh, "mkdir -p /home/pi/asl")
    sftp = ssh.open_sftp()
    sftp.put(os.path.join(HERE, "pi_device.py"), "/home/pi/asl/pi_device.py")
    with sftp.open("/tmp/asl-device.service", "w") as f:
        f.write(SERVICE)
    sftp.close()
    print("[deploy] files copied")

    if run(ssh, "python3 -c 'import serial'", show=False)[0] != 0:
        print("[deploy] installing python3-serial ...")
        run(ssh, "apt-get install -y python3-serial", sudo=True, show=False)
    run(ssh, "for c in preconfigured HAWK; do nmcli connection modify $c 802-11-wireless.powersave 2 2>/dev/null; done; "
             "iw dev wlan0 set power_save off 2>/dev/null; true", sudo=True, show=False)
    run(ssh, "cp /tmp/asl-device.service /etc/systemd/system/asl-device.service && systemctl daemon-reload "
             "&& systemctl enable asl-device && systemctl restart asl-device", sudo=True)
    import time
    time.sleep(6)
    run(ssh, "systemctl is-active asl-device; journalctl -u asl-device -n 15 --no-pager -o cat")
    ssh.close()


if __name__ == "__main__":
    main()
