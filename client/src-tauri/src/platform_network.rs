//! Account/control traffic uses the bundled transport shared with the provider
//! and overflow. The helper resolves live OS proxy/PAC policy and uses OS trust.
//! There is no credential in its command line and no fallback to another stack.
use std::io::{Read, Write};
use std::process::{Command, Stdio};

pub(crate) fn validate_url(url: &str) -> Result<String, String> {
    let parsed = reqwest::Url::parse(url).map_err(|_| "invalid platform URL")?;
    let host = parsed
        .host_str()
        .unwrap_or("")
        .trim_start_matches('[')
        .trim_end_matches(']');
    let local = host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .map(|ip| ip.is_loopback())
            .unwrap_or(false);
    if !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.fragment().is_some()
        || !(parsed.scheme() == "https" || (parsed.scheme() == "http" && local))
    {
        return Err("platform URL must use HTTPS (HTTP is allowed only on loopback)".into());
    }
    Ok(parsed.to_string())
}

fn decode(output: &[u8]) -> Result<super::PlatformResponse, String> {
    let split = output
        .iter()
        .position(|b| *b == b'\n')
        .ok_or("incomplete network helper reply")?;
    let head: serde_json::Value =
        serde_json::from_slice(&output[..split]).map_err(|_| "invalid network helper reply")?;
    if let Some(error) = head.get("error").and_then(|v| v.as_str()) {
        return Err(error.to_string());
    }
    let status = head
        .get("status")
        .and_then(|v| v.as_u64())
        .filter(|s| (100..=599).contains(s))
        .ok_or("missing HTTP status from network helper")?;
    let body = String::from_utf8(output[split + 1..].to_vec())
        .map_err(|_| "platform reply is not UTF-8")?;
    Ok(super::PlatformResponse {
        status: status as u16,
        body,
    })
}

pub(crate) async fn request(
    command: Command,
    method: String,
    url: String,
    body: Option<String>,
    bearer: Option<String>,
    timeout_ms: Option<u64>,
) -> Result<super::PlatformResponse, String> {
    let timeout_ms = timeout_ms.unwrap_or(20_000).clamp(1, 120_000);
    let payload = serde_json::to_vec(&serde_json::json!({
        "method": method, "url": url, "body": body, "bearer": bearer, "timeout_ms": timeout_ms,
    }))
    .map_err(|_| "cannot encode platform request")?;
    if payload.len() > 64 * 1024 * 1024 {
        return Err("platform request exceeds byte limit".into());
    }
    let mut framed = (payload.len() as u32).to_be_bytes().to_vec();
    framed.extend_from_slice(&payload);
    tauri::async_runtime::spawn_blocking(move || {
        let mut command = command;
        command
            .arg("--platform-http-stdio")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        let mut child = command
            .spawn()
            .map_err(|_| "could not start bundled network helper")?;
        let mut stdin = child.stdin.take().ok_or("missing helper input pipe")?;
        let stdout = child.stdout.take().ok_or("missing helper output pipe")?;
        let reader = std::thread::spawn(move || {
            let mut bytes = Vec::new();
            stdout
                .take(64 * 1024 * 1024 + 2049)
                .read_to_end(&mut bytes)
                .map(|_| bytes)
        });
        // All helper input is framed, so neither side waits for EOF to start.
        if stdin.write_all(&framed).is_err() {
            let _ = child.kill();
            let _ = child.wait();
            let _ = reader.join();
            return Err("could not send request to network helper".into());
        }
        drop(stdin);
        let deadline =
            std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms + 1500);
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) if std::time::Instant::now() < deadline => {
                    std::thread::sleep(std::time::Duration::from_millis(20))
                }
                _ => {
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
            }
        };
        let output = reader
            .join()
            .map_err(|_| "network helper reader failed")?
            .map_err(|_| "network helper disconnected")?;
        if output.len() > 64 * 1024 * 1024 + 2048 {
            return Err("platform reply exceeds byte limit".into());
        }
        match status {
            Some(status) if status.success() => decode(&output),
            Some(_) => Err(decode(&output)
                .err()
                .unwrap_or_else(|| "network helper exited unexpectedly".into())),
            None => Err("platform request timed out".into()),
        }
    })
    .await
    .map_err(|_| "network helper task failed".to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_https_and_both_loopback_families_only() {
        for url in [
            "https://example.test:8443/api?a=1",
            "http://127.0.0.2/a",
            "http://[::1]/a",
        ] {
            assert!(validate_url(url).is_ok(), "{url}");
        }
        for url in [
            "http://example.test/",
            "https://user:secret@example.test/",
            "file:///tmp/x",
        ] {
            assert!(validate_url(url).is_err(), "{url}");
        }
    }

    #[test]
    fn keeps_http_errors_distinct_from_transport_errors() {
        let reply = decode(b"{\"status\":401}\n{\"error\":\"invalid credentials\"}").unwrap();
        assert_eq!(reply.status, 401);
        assert_eq!(reply.body, "{\"error\":\"invalid credentials\"}");
        assert!(decode(b"{\"error\":\"certificate verification failed\"}\n").is_err());
        assert!(decode(b"{\"status\":200}").is_err());
    }

    #[test]
    #[ignore = "requires the compiled platform agent; run explicitly after make -f Makefile.platform"]
    fn native_command_runs_real_shared_transport() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/control", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(std::time::Duration::from_secs(3)))
                .unwrap();
            let mut bytes = Vec::new();
            loop {
                let mut chunk = [0; 1024];
                let count = socket.read(&mut chunk).unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&chunk[..count]);
                if bytes.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            let headers = String::from_utf8(bytes).unwrap();
            assert!(headers.contains(&format!(
                "X-IdleToken-Version: {}",
                env!("CARGO_PKG_VERSION")
            )));
            assert!(headers.contains("Authorization: Bearer pipe-only-test-token"));
            socket.write_all(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").unwrap();
        });
        let binary = std::env::var("IDLETOKEN_AGENT_TEST_BIN").unwrap_or_else(|_| {
            format!(
                "{}/../../build/idletoken-platform-agent",
                env!("CARGO_MANIFEST_DIR")
            )
        });
        let result = tauri::async_runtime::block_on(request(
            Command::new(binary),
            "GET".into(),
            url,
            None,
            Some("pipe-only-test-token".into()),
            Some(3000),
        ))
        .unwrap();
        assert_eq!(result.status, 401);
        assert_eq!(result.body, "{}");
        server.join().unwrap();
    }
}
