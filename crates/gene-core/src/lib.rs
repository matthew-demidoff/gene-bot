//! gene-core: the engine behind gene. The LLM streaming client and incremental
//! `<think>`/```run parser, the agentic shell tools, the conversation + dataset
//! model, persistence, configuration, inference providers, and the MLX LoRA
//! training pipeline.
//!
//! This crate is UI-agnostic — frontends (the `gene` CLI, the egui GUI) build on
//! top of it and share all logic through these modules.

pub mod chat;
pub mod config;
pub mod doctor;
pub mod llm;
pub mod model;
pub mod persist;
pub mod provider;
pub mod runs;
pub mod tools;
pub mod train;
