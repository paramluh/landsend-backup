BEGIN TRANSACTION;
CREATE TABLE _sqlx_migrations (
    version BIGINT PRIMARY KEY,
    description TEXT NOT NULL,
    installed_on TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    success BOOLEAN NOT NULL,
    checksum BLOB NOT NULL,
    execution_time BIGINT NOT NULL
);
INSERT INTO "_sqlx_migrations" VALUES(1,'memories','2026-10-27 21:37:22',1,X'A1AF50DA50775A70F98DA680006B7DF4501956753AE98511AB727AA295915D9AE6F4190B4EAA5ABC4FD7891642F7D21D',378415);
INSERT INTO "_sqlx_migrations" VALUES(2,'consolidation progress','2026-10-27 21:37:22',1,X'18F0A8DD7FE9A847B30D719029066D7A78E0BC64310DC66E4F7708BAD6F1A0A594C0DBD14FEC1CCB410CD7AE75DD7B10',179487);
CREATE TABLE consolidation_progress (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    max_thread_count INTEGER NOT NULL DEFAULT 0
);
INSERT INTO "consolidation_progress" VALUES(1,0);
CREATE TABLE jobs (
    kind TEXT NOT NULL,
    job_key TEXT NOT NULL,
    status TEXT NOT NULL,
    worker_id TEXT,
    ownership_token TEXT,
    started_at INTEGER,
    finished_at INTEGER,
    lease_until INTEGER,
    retry_at INTEGER,
    retry_remaining INTEGER NOT NULL,
    last_error TEXT,
    input_watermark INTEGER,
    last_success_watermark INTEGER,
    PRIMARY KEY (kind, job_key)
);
CREATE TABLE stage1_outputs (
    thread_id TEXT PRIMARY KEY,
    source_updated_at INTEGER NOT NULL,
    raw_memory TEXT NOT NULL,
    rollout_summary TEXT NOT NULL,
    rollout_slug TEXT,
    generated_at INTEGER NOT NULL,
    usage_count INTEGER,
    last_usage INTEGER,
    selected_for_phase2 INTEGER NOT NULL DEFAULT 0,
    selected_for_phase2_source_updated_at INTEGER
);
CREATE INDEX idx_stage1_outputs_source_updated_at
    ON stage1_outputs(source_updated_at DESC, thread_id DESC);
CREATE INDEX idx_jobs_kind_status_retry_lease
    ON jobs(kind, status, retry_at, lease_until);
COMMIT;
