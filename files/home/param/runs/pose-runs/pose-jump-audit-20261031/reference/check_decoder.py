import sys,json,numpy as np,torch
from pathlib import Path
sys.path.insert(0,'/home/param/websites/pose-renderer/bridge/engine/posetok')
from pose_tokenizer import PoseTokenizer
out=Path(__file__).parent
torch.set_num_threads(4)
tok=PoseTokenizer.from_pretrained('/home/param/models/pose-tokenizer-32cb-200k-noshift',device='cpu')
codes=np.load(out/'tokens.npz')['pose']
def decode(a):
 with torch.inference_mode():
  return tok.decode([torch.from_numpy(row.copy()).long()[None] for row in a.T]).float()[0].numpy().reshape(-1,55,3)
full=decode(np.concatenate([codes,np.repeat(codes[-1:],2,axis=0)]))
unpadded=decode(codes)
prefix=decode(codes[:272])
shifted=decode(codes[64:])
checks={
 'unpadded_error_at_jump':float(np.max(np.abs(full[508:516]-unpadded[508:516]))),
 'prefix272_error_at_jump':float(np.max(np.abs(full[508:516]-prefix[508:516]))),
 'shifted64codeframes_error_at_jump':float(np.max(np.abs(full[508:516]-shifted[380:388]))),
 'full_feature_step_511_512_l2':float(np.linalg.norm(full[512]-full[511])),
 'shifted_feature_step_383_384_l2':float(np.linalg.norm(shifted[384]-shifted[383])),
 'same_source_jump_at_new_frame':384,
 'raw_token_frame':256,
 'changed_codebooks_at_256':int(np.count_nonzero(codes[256]!=codes[255])),
 'raw_first_8_codes_at_255':codes[255,:8].tolist(),
 'raw_first_8_codes_at_256':codes[256,:8].tolist(),
 'conclusion':'Jump survives removing tail padding, truncating sequence after event and moving event away from decoder frame512. Not a decoder512 chunk/reset boundary artifact.',
}
np.savez_compressed(out/'reference-decoded.npz',features=full[:len(codes)*2])
(out/'decoder-boundary-check.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
