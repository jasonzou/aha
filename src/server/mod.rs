use axum::{
    routing::{get, post},
    Router,
};
use std::net::IpAddr;
use std::str::FromStr;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tower_http::cors::CorsLayer;

pub(crate) mod api;
pub(crate) mod asr;
pub(crate) mod embedding;
pub(crate) mod process;
pub(crate) mod reranker;

pub(crate) async fn start_http_server(
    address: String,
    port: u16,
    allow_remote_shutdown: bool,
) -> anyhow::Result<()> {
    api::set_server_port(port, allow_remote_shutdown);

    let pid = std::process::id();
    process::create_pid_file(pid, port)?;

    let (shutdown_tx, mut shutdown_rx) = oneshot::channel::<()>();

    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        println!("Received shutdown signal, gracefully shutting down...");
        let _ = shutdown_tx.send(());
    });

    let ip_address: IpAddr = IpAddr::from_str(&address)?;

    let cors = CorsLayer::permissive();

    let app = Router::new()
        .route("/v1/chat/completions", post(api::chat))
        .route("/chat/completions", post(api::chat))
        .route("/images/remove_background", post(api::remove_background))
        .route("/audio/speech", post(api::speech))
        .route("/audio/transcriptions", post(asr::transcriptions))
        .route("/v1/audio/transcriptions", post(asr::transcriptions))
        .route("/embeddings", post(embedding::embeddings))
        .route("/v1/embeddings", post(embedding::embeddings))
        .route("/rerank", post(reranker::rerank))
        .route("/v1/rerank", post(reranker::rerank))
        .route("/health", get(api::health))
        .route("/models", get(api::models))
        .route("/v1/models", get(api::models))
        .route("/shutdown", post(api::shutdown))
        .layer(cors);

    let addr = (ip_address, port);
    let listener = TcpListener::bind(addr).await?;
    println!("Server listening on {}", addr.1);

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            shutdown_rx.await.ok();
        })
        .await?;

    process::cleanup_pid_file(port)?;

    Ok(())
}