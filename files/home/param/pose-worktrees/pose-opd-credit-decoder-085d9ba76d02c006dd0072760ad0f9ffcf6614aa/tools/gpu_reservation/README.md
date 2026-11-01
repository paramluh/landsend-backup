# GPU reservation

The user reserved `landsend-1` through `landsend-4` exclusively for `param` work
on October 28, 2026. The later four-node instruction supersedes the earlier
reservation that included node 0: **do not kill processes on landsend-0**.
Its guard is stopped and disabled. This source also refuses to run on node 0.
The guard is hard-limited to nodes 1–4 and UID 1003 (`param`). It does not run
on landsend-5 or landsend-12.

Every three seconds, the root service checks NVIDIA's compute-process list.
Foreign non-root compute processes receive SIGTERM, then SIGKILL after a
three-second grace period if they still hold GPU contexts. Both UID and process
start time are rechecked; Linux pidfds prevent signaling a reused PID. The
guard does not kill CPU-only launchers or alter GPU drivers/device permissions.
Root compute processes and NVIDIA infrastructure daemons are logged for manual
review, not automatically killed. Administrative/root bypass cannot be ruled
out; this is a reactive GPU-job guard, not an access-control boundary.

Installed files on each reserved host:

- `/usr/local/lib/param-gpu-reservation/guard.py` (root-owned copy)
- `/etc/systemd/system/param-gpu-reservation.service`
- `/var/lib/param-gpu-reservation/status.json` (heartbeat and counters)

Inspect or stop the reservation on one host:

```bash
sudo systemctl status param-gpu-reservation.service
cat /var/lib/param-gpu-reservation/status.json
sudo journalctl -u param-gpu-reservation.service --since today
sudo systemctl disable --now param-gpu-reservation.service
```

Disabling the service restores the previous unrestricted scheduling behavior;
no device-mode or driver settings need restoring. The source can be audited
without sending any signals:

```bash
sudo python3 tools/gpu_reservation/guard.py --audit-once
```
