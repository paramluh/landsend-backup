"""Poll owned benchmark progress and sample whole-device memory across all ranks."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from common import HOSTS, remote, write_json
from status import script

BASE = Path(__file__).resolve().parent
EVIDENCE = BASE / 'snapshot-1793450446728000042' / 'runtime'


def main():
    EVIDENCE.mkdir(exist_ok=True)
    peaks = {host: {} for host in HOSTS}
    signatures = {}
    started = time.time()
    samples = 0
    with (EVIDENCE / 'status-samples.jsonl').open('a', buffering=1) as stream:
        while True:
            with ThreadPoolExecutor(max_workers=5) as pool:
                nodes = list(pool.map(lambda host: json.loads(remote(host, ['python3', '-B', '-c', script]).stdout), HOSTS))
            now = time.time()
            stream.write(json.dumps({'time': now, 'nodes': nodes}) + '\n')
            samples += 1
            for node in nodes:
                host = node['host']
                for row in node['gpu_usage']:
                    index, uuid, used, total, util = [value.strip() for value in row.split(',')]
                    record = peaks[host].setdefault(index, {'uuid': uuid, 'peak_used_mib': 0, 'total_mib': int(total)})
                    if int(used) > record['peak_used_mib']:
                        record.update(peak_used_mib=int(used), time=now)
                phases = sorted({event.get('phase', event['event']) for event in node['rank_last_events'].values()})
                finished = node.get('finished')
                signature = (tuple(phases), json.dumps(finished), node['torchrun']['running'])
                if signatures.get(host) != signature:
                    print(json.dumps({'time': now, 'host': host, 'phases': phases, 'finished': finished,
                                      'torchrun_running': node['torchrun']['running'],
                                      'peak_memory_gib': round(max(v['peak_used_mib'] for v in peaks[host].values()) / 1024, 3)}), flush=True)
                    signatures[host] = signature
                if any(not process['owned'] for process in node['gpu_compute_processes']):
                    print(json.dumps({'unexpected_gpu_process': node['gpu_compute_processes'], 'host': host}), flush=True)
            write_json(EVIDENCE / 'latest.json', {'time': now, 'nodes': nodes})
            write_json(EVIDENCE / 'memory-peaks.json', {'sampling_interval_seconds': 5,
                       'scope': 'whole-device NVML used memory including all compute processes; sampled lower bound on instantaneous peak',
                       'samples': samples, 'nodes': peaks})
            if any(node.get('finished', {}).get('exit_code', 0) != 0 for node in nodes):
                print(json.dumps({'failure': True, 'finished': {n['host']: n.get('finished') for n in nodes}}), flush=True)
            if all('finished' in node for node in nodes):
                write_json(EVIDENCE / 'completion.json', {'started_monitoring': started, 'completed': now, 'nodes': nodes, 'peaks': peaks})
                print(json.dumps({'complete': True, 'exit_codes': {node['host']: node['finished']['exit_code'] for node in nodes}}), flush=True)
                return
            if now - started > 3600:
                raise TimeoutError('Monitoring exceeded one hour; no process was signalled')
            time.sleep(5)


if __name__ == '__main__':
    main()
