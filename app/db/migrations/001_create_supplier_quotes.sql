CREATE TABLE IF NOT EXISTS supplier_quotes (
id BIGSERIAL PRIMARY KEY,
sku TEXT NOT NULL,
site TEXT NOT NULL,
region TEXT,
url TEXT NOT NULL,
title TEXT NOT NULL,
currency TEXT NOT NULL,
price_raw BIGINT NOT NULL,
fx_to TEXT,
price_jpy BIGINT,
terms TEXT,
notes JSONB,
fetched_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_sq_sku_time ON supplier_quotes (sku, fetched_at DESC);
CREATE INDEX IF NOT EXISTS idx_sq_sku_site_time ON supplier_quotes (sku, site, fetched_at DESC);
CREATE INDEX IF NOT EXISTS idx_sq_site_time ON supplier_quotes (site, fetched_at DESC);