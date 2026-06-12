-- AskDesk schema. Run once against your Postgres (Supabase SQL editor works).
-- Requires the pgvector extension (preinstalled on Supabase).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source    TEXT NOT NULL,
    ordinal   INT  NOT NULL,
    text      TEXT NOT NULL,
    -- dimension must match ASKDESK_EMBED_DIM (1536 = text-embedding-3-small)
    embedding VECTOR(1536) NOT NULL,
    tsv       TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (source, ordinal)
);

-- ANN index for vector search (cosine). Rebuild with more lists as corpus grows.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- GIN index for the keyword half of hybrid retrieval.
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
