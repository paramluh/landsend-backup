BEGIN TRANSACTION;
CREATE TABLE _sqlx_migrations (
    version BIGINT PRIMARY KEY,
    description TEXT NOT NULL,
    installed_on TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    success BOOLEAN NOT NULL,
    checksum BLOB NOT NULL,
    execution_time BIGINT NOT NULL
);
INSERT INTO "_sqlx_migrations" VALUES(1,'thread goals','2026-10-27 21:37:22',1,X'8CE6280244A0B4B39C7C37C7F67AFF1670EC963D2401CDF2ABCFEA9904102DB6CD99B7BEDF366267791A45EB12F17713',236106);
INSERT INTO "_sqlx_migrations" VALUES(2,'thread goal continuation deferrals','2026-10-27 21:37:22',1,X'F3FF7DE2FD89914820BEF371081E885FD17DF18A6FBDC432399A83255A8A01AA32FB7EEA5E4DE21A17BDA3DA16C49584',189577);
CREATE TABLE thread_goal_continuation_deferrals (
    thread_id TEXT PRIMARY KEY NOT NULL REFERENCES thread_goals(thread_id) ON DELETE CASCADE
);
CREATE TABLE thread_goals (
    thread_id TEXT PRIMARY KEY NOT NULL,
    goal_id TEXT NOT NULL,
    objective TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN (
        'active',
        'paused',
        'blocked',
        'usage_limited',
        'budget_limited',
        'complete'
    )),
    token_budget INTEGER,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    time_used_seconds INTEGER NOT NULL DEFAULT 0,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL
);
INSERT INTO "thread_goals" VALUES('01a18ea9-873a-7643-ad2e-1c2026338760','13b7ef53-2761-4e50-adc9-23dd6b603175','Wait actually stop this. Here''s what I want you to do: please use the three nodes to run a finetune of the original aloe on *only* black bands. Then run inference on 20 clips. By only black bands, I mean I want you to pick only the dataset rows which have at least one black band section. Train on 1min clips which include the black bands and batch this up to 10mins (10 1 min clips per sequence).&#x20;
set this up and get training super fast. No need to log on wandb and just save the final checkpoint.','paused',NULL,207105,307,1793417386640,1793418873243);
INSERT INTO "thread_goals" VALUES('01a1949b-cd8c-7b83-a918-894ac1c9824a','fff68213-79c4-4edc-9577-c308c4ee7d6a','Before cluster access expires at midnight San Francisco time, back up checkpoints not already on Hugging Face; commit and push unfinished work in each home repository to tmp/landsend-backup-param; back up code and other non-repository project material to a new private landsend-backup repository in the user''s personal GitHub account, verify backups, and report any gaps.','active',NULL,322701,529,1793505708608,1793506237978);
COMMIT;
