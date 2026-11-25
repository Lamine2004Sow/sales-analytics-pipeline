-- Runs once, when the warehouse volume is first created (docker-entrypoint-initdb.d).
-- Metabase keeps its own metadata in a separate database so it never mixes with the data.
CREATE DATABASE metabase_app OWNER retail;

\c retail

-- Star schema and marts written by `retail-pipeline load`.
CREATE SCHEMA IF NOT EXISTS analytics AUTHORIZATION retail;
