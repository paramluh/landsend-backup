"""one-off: the live Aloe demo's held-out persona presets (aloe-m1vox-live/prompts_ppm.Presets: held-out Aloe rows, named B personas) ->
control/personas.json [{id, label, name, persona}] for the server's persona picker."""
import json
import sys
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-m1vox-live")
import prompts_ppm as PR  # noqa: E402
p = PR.Presets(n_max=24)
out = [{"id": x["id"], "label": x["label"], "name": x["name"], "persona": x["persona"]} for x in p.items]
json.dump(out, open("/data/home/amartyavarma/ppm/ppm-pose-server/control/personas.json", "w"), indent=1)
print(len(out), [x["label"] for x in out][:5])
