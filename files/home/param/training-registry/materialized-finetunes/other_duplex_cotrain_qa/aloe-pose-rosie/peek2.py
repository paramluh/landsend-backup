import pyarrow.parquet as pq
t=pq.read_table("/data/home/amartyavarma/ppm/aloe-pose-ft/stores/p0/rows.parquet").to_pydict()
for k in ('user_speaker_dir','user_speaker','assistant_speaker_dir','assistant_speaker','src_conversation','src_dir','frame_start','frame_end','chunk_start_sec','n_chunks_conv','xbleed','b_ok','split'):
    print(k, t[k][:3])
