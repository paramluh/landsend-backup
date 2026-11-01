"""Voice presets from actual raw-code clips shipped with this checkpoint.

The original full training-store catalog is not required for inference. Keep the
same prompt builder, and validate that every packaged clip enters its B stream
unchanged. Package metadata preserves the source recording and clip window.
"""
import json
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file
import prompts_ppm as PR


class PackagedPresets(PR.Presets):
    def __init__(self, root, codes='raw', first=None, extra=None):
        self.codes, self.first, self.extra = codes, first, extra
        self.alias, self.items = {}, []
        for directory in sorted(Path(root).iterdir()):
            clip_file = directory / 'voice_clip.safetensors'
            if not clip_file.is_file():
                continue
            values = np.asarray(load_file(clip_file)['codes'], dtype=np.int64)
            assert values.ndim == 2 and values.shape[0] == 32
            assert 38 <= values.shape[1] <= 125 and values.min() >= 0 and values.max() < 2048
            metadata = json.loads((directory / 'meta.json').read_text())
            clip = PR.upload_clip(values)
            clip.update(src='packaged_raw', src_key=f'OPENINGS:openings-raw/{directory.name}/voice_clip', src_file=str(clip_file))
            label = 'Person ' + directory.name[:2] if directory.name[:2].isdigit() else 'Raw voice ' + directory.name[4:12]
            self.items.append({'id':directory.name, 'name':label, 'label':label,
                               'persona':'', 'prose':'', 'clip':clip,
                               'row_key':metadata.get('sources',{}).get('voice_clip',{}).get('src_row_key',directory.name),
                               'group':'Checkpoint raw voice clips'})
        assert self.items, f'No packaged raw voice clips in {root}'
        if first:
            assert self.get(first), f'Missing default packaged voice {first}'
            self.items.sort(key=lambda p:p['id'] != first)

    def selftest(self, n=None):
        checked=[]
        for preset in self.items:
            clip=preset['clip']
            prefix,_=PR.build_prefix('',clip)
            expected=np.asarray(clip['src_row']['codes_user'],np.int64)
            depth=np.asarray(prefix['depth'])
            active=(depth[1]>=0).all(axis=-1)
            actual=depth[1,active]-np.arange(32)*2048
            equal=np.array_equal(actual,expected.T)
            assert equal and (depth[0]<0).all(), preset['id']
            checked.append({'preset':preset['id'],'frames':clip['L'],'P':int(prefix['P']),'unchanged_B_codes':equal})
        empty,_=PR.build_prefix('',None)
        assert empty['P']==5 and (np.asarray(empty['depth'])<0).all()
        return {'PASS':True,'test':'packaged raw clip integrity and unchanged B prefix codes; empty prefix has P=5','clips':checked}
