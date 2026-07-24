//! Remote filesystem implementation
//!
//! Implements the FileSystem trait for remote operations via SSH agent.

use crate::model::filesystem::{
    DirEntry, EntryType, FileMetadata, FilePermissions, FileReader, FileSystem, FileWriter, WriteOp,
};
use crate::services::remote::channel::{AgentChannel, ChannelError};
use crate::services::remote::protocol::{
    append_params, count_lf_params, decode_base64, ls_params, patch_params, read_params,
    stat_params, sudo_write_params, truncate_params, write_params, PatchOp, RemoteDirEntry,
    RemoteMetadata,
};
use std::io::{self, Cursor, Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, UNIX_EPOCH};

/// Static per-connection system info fetched once from the agent's `info`
/// response. `$HOME` and the temp dir don't change over a session, so they are
/// cached — the sync `FileSystem` accessors that need them
/// ([`RemoteFileSystem::home_dir`] / [`RemoteFileSystem::unique_temp_path`])
/// run on the editor thread, and issuing a blocking request there would hang
/// the whole single-threaded UI for the full request timeout when the link is
/// unresponsive (issue: web editor freezes on switching to an unreachable SSH
/// workspace).
#[derive(Debug, Clone)]
struct RemoteSysInfo {
    /// Remote `$HOME`, when the agent reported one.
    home: Option<PathBuf>,
    /// Remote temp dir (agent's `tempfile.gettempdir()`), `/tmp` fallback.
    temp_dir: PathBuf,
}

/// A short-lived read cache used to materialize a freshly-connected session's
/// workspace without blocking the editor thread. Populated off the hot path by
/// [`RemoteFileSystem::prewarm_paths_async`] (on the connect worker) and
/// consulted by the read/stat/canonicalize accessors, then dropped by
/// `clear_prewarm` once the restore has consumed it. Empty in the steady state,
/// so it adds only a lock + miss to every op while warm and nothing once
/// cleared.
#[derive(Default)]
struct PrewarmCache {
    /// Full file content, keyed by both the requested and canonical path.
    content: std::collections::HashMap<PathBuf, Vec<u8>>,
    /// `metadata` results (follow-symlink stat).
    meta: std::collections::HashMap<PathBuf, FileMetadata>,
    /// `canonicalize` results.
    canon: std::collections::HashMap<PathBuf, PathBuf>,
}

/// Remote filesystem that communicates with the Python agent
pub struct RemoteFileSystem {
    channel: Arc<AgentChannel>,
    /// Display string for the connection
    connection_string: String,
    /// Cached `info` snapshot (see [`RemoteSysInfo`]). Primed once on the
    /// connect worker by [`RemoteFileSystem::prime_sys_info`]; read lock-free
    /// thereafter so the editor-thread accessors that need `$HOME` / the temp
    /// dir don't issue a blocking round-trip on the hot path.
    sys_info: Arc<OnceLock<RemoteSysInfo>>,
    /// Short-lived materialization read cache (see [`PrewarmCache`]). Wrapped in
    /// a `Mutex` because the connect worker populates it while the editor thread
    /// may read it; empty except during the window between a dive's prewarm and
    /// its restore.
    prewarm: Arc<std::sync::Mutex<PrewarmCache>>,
}

impl RemoteFileSystem {
    /// Create a new remote filesystem from an agent channel
    pub fn new(channel: Arc<AgentChannel>, connection_string: String) -> Self {
        Self {
            channel,
            connection_string,
            sys_info: Arc::new(OnceLock::new()),
            prewarm: Arc::new(std::sync::Mutex::new(PrewarmCache::default())),
        }
    }

    /// Parse the agent `info` response into the cached [`RemoteSysInfo`].
    fn parse_sys_info(info: &serde_json::Value) -> RemoteSysInfo {
        RemoteSysInfo {
            home: info.get("home").and_then(|v| v.as_str()).map(PathBuf::from),
            temp_dir: Self::parse_temp_dir_from_info(Some(info)),
        }
    }

    /// Fetch the static `info` (home / temp dir) once and cache it, awaiting
    /// the agent's reply. Call this on the **connect worker** (never the editor
    /// thread): it doubles as a liveness gate — a host that completed the SSH
    /// handshake but can't answer the agent (a stalled/half-open link) makes
    /// this error, letting the caller refuse to promote a session that would
    /// otherwise block the editor thread on the request timeout for every
    /// subsequent file op. Once it succeeds, `home_dir` / `unique_temp_path`
    /// serve from the cache and never touch the link again.
    pub async fn prime_sys_info(&self) -> io::Result<()> {
        if self.sys_info.get().is_some() {
            return Ok(());
        }
        let resp = self
            .channel
            .request("info", serde_json::json!({}))
            .await
            .map_err(Self::to_io_error)?;
        // A concurrent prime may have set it first; either snapshot is fine.
        #[allow(clippy::let_underscore_must_use)]
        let _ = self.sys_info.set(Self::parse_sys_info(&resp));
        Ok(())
    }

    /// The connection's static `info` (home / temp dir): the value cached at
    /// connect time when present, else a one-shot blocking fetch that caches
    /// its result. On the SSH connect path `prime_sys_info` primes this ahead
    /// of time, so the editor-thread callers (`home_dir` during workspace
    /// restore / file open, the file-explorer root) hit the cache and never
    /// block. The blocking branch is only reached by a directly-constructed
    /// filesystem (the `--remote` CLI startup, tests), where a synchronous
    /// resolve is expected and there is no editor loop to stall.
    fn sys_info(&self) -> Option<RemoteSysInfo> {
        if let Some(info) = self.sys_info.get() {
            return Some(info.clone());
        }
        let resp = self
            .channel
            .request_blocking("info", serde_json::json!({}))
            .ok()?;
        let info = Self::parse_sys_info(&resp);
        // A concurrent prime may have set it first; either snapshot is fine.
        #[allow(clippy::let_underscore_must_use)]
        let _ = self.sys_info.set(info.clone());
        Some(info)
    }

    /// Prewarmed full content for `path`, if the materialization cache holds it.
    fn prewarm_content(&self, path: &Path) -> Option<Vec<u8>> {
        self.prewarm.lock().unwrap().content.get(path).cloned()
    }

    /// Prewarmed `metadata` result for `path`, if cached.
    fn prewarm_meta(&self, path: &Path) -> Option<FileMetadata> {
        self.prewarm.lock().unwrap().meta.get(path).cloned()
    }

    /// Prewarmed `canonicalize` result for `path`, if cached.
    fn prewarm_canon(&self, path: &Path) -> Option<PathBuf> {
        self.prewarm.lock().unwrap().canon.get(path).cloned()
    }

    /// Warm the read cache for `paths` — a freshly-connected session's persisted
    /// buffers — by fetching each one's canonical form, metadata and full
    /// content **asynchronously**, so a later editor-thread restore serves them
    /// from memory instead of blocking on the (possibly slow) link.
    ///
    /// Awaited on the connect worker's runtime (inside `connect_ssh_authority`),
    /// *before* the authority is handed to the editor — so it uses the async
    /// channel API (never `block_on`, which would panic here) and the editor
    /// loop is never involved. Each result is stashed under both the requested
    /// and the canonical path, matching how the restore canonicalizes then stats
    /// and reads. Best-effort: a path that errors is simply left uncached, and
    /// the restore falls through to a live read that surfaces the real error.
    pub async fn prewarm_paths_async(&self, paths: &[PathBuf]) {
        for path in paths {
            let path_str = path.to_string_lossy().to_string();

            let canon = self
                .channel
                .request("realpath", serde_json::json!({ "path": path_str }))
                .await
                .ok()
                .and_then(|r| r.get("path").and_then(|v| v.as_str()).map(PathBuf::from));
            let keys: Vec<PathBuf> = match &canon {
                Some(c) if c != path => vec![path.clone(), c.clone()],
                _ => vec![path.clone()],
            };
            if let Some(c) = &canon {
                self.prewarm
                    .lock()
                    .unwrap()
                    .canon
                    .insert(path.clone(), c.clone());
            }

            if let Ok(result) = self
                .channel
                .request("stat", stat_params(&path_str, true))
                .await
            {
                if let Ok(rm) = serde_json::from_value::<RemoteMetadata>(result) {
                    let name = path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    let meta = Self::convert_metadata(&rm, &name);
                    let mut cache = self.prewarm.lock().unwrap();
                    for k in &keys {
                        cache.meta.insert(k.clone(), meta.clone());
                    }
                }
            }

            if let Ok((chunks, _result)) = self
                .channel
                .request_with_data("read", read_params(&path_str, None, None))
                .await
            {
                let mut content = Vec::new();
                for chunk in chunks {
                    if let Some(b64) = chunk.get("data").and_then(|v| v.as_str()) {
                        if let Ok(decoded) = decode_base64(b64) {
                            content.extend(decoded);
                        }
                    }
                }
                let mut cache = self.prewarm.lock().unwrap();
                for k in &keys {
                    cache.content.insert(k.clone(), content.clone());
                }
            }
        }
    }

    /// Get the connection string for display
    pub fn connection_string(&self) -> &str {
        &self.connection_string
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.channel.is_connected()
    }

    /// Extract the remote temp directory from an agent `info` response.
    /// Falls back to `/tmp` if the response is missing or doesn't contain `temp_dir`.
    fn parse_temp_dir_from_info(info: Option<&serde_json::Value>) -> PathBuf {
        info.and_then(|r| {
            r.get("temp_dir")
                .and_then(|v| v.as_str())
                .map(PathBuf::from)
        })
        .unwrap_or_else(|| PathBuf::from("/tmp"))
    }

    /// Convert a ChannelError to io::Error
    fn to_io_error(e: ChannelError) -> io::Error {
        match e {
            ChannelError::Io(e) => e,
            ChannelError::Remote(msg) => {
                let kind = if msg.contains("not found") || msg.contains("No such file") {
                    io::ErrorKind::NotFound
                } else if msg.contains("permission denied") {
                    io::ErrorKind::PermissionDenied
                } else if msg.contains("is a directory") {
                    io::ErrorKind::IsADirectory
                } else if msg.contains("not a directory") {
                    io::ErrorKind::NotADirectory
                } else {
                    io::ErrorKind::Other
                };
                io::Error::new(kind, msg)
            }
            e => io::Error::other(e.to_string()),
        }
    }

    /// Convert remote metadata to FileMetadata
    fn convert_metadata(rm: &RemoteMetadata, name: &str) -> FileMetadata {
        let modified = if rm.mtime > 0 {
            Some(UNIX_EPOCH + Duration::from_secs(rm.mtime as u64))
        } else {
            None
        };

        let is_hidden = name.starts_with('.');
        let permissions = FilePermissions::from_mode(rm.mode);

        #[cfg(unix)]
        let is_readonly = {
            let (euid, user_groups) =
                crate::model::filesystem::StdFileSystem::current_user_groups();
            permissions.is_readonly_for_user(euid, rm.uid, rm.gid, &user_groups)
        };
        #[cfg(not(unix))]
        let is_readonly = permissions.is_readonly();

        let mut meta = FileMetadata::new(rm.size)
            .with_hidden(is_hidden)
            .with_readonly(is_readonly)
            .with_permissions(permissions);

        if let Some(m) = modified {
            meta = meta.with_modified(m);
        }

        #[cfg(unix)]
        {
            meta.uid = Some(rm.uid);
            meta.gid = Some(rm.gid);
        }

        meta
    }

    /// Convert remote dir entry to DirEntry
    fn convert_dir_entry(re: &RemoteDirEntry) -> DirEntry {
        let entry_type = if re.link {
            EntryType::Symlink
        } else if re.dir {
            EntryType::Directory
        } else {
            EntryType::File
        };

        let modified = if re.mtime > 0 {
            Some(UNIX_EPOCH + Duration::from_secs(re.mtime as u64))
        } else {
            None
        };

        let is_hidden = re.name.starts_with('.');
        let permissions = FilePermissions::from_mode(re.mode);
        let is_readonly = permissions.is_readonly();

        let mut metadata = FileMetadata::new(re.size)
            .with_hidden(is_hidden)
            .with_readonly(is_readonly)
            .with_permissions(permissions);

        if let Some(m) = modified {
            metadata = metadata.with_modified(m);
        }

        let mut entry = DirEntry::new(PathBuf::from(&re.path), re.name.clone(), entry_type);
        entry.metadata = Some(metadata);
        entry.symlink_target_is_dir = re.link_dir;

        entry
    }
}

impl FileSystem for RemoteFileSystem {
    fn read_file(&self, path: &Path) -> io::Result<Vec<u8>> {
        // Serve from the materialization prewarm cache when warm, so reopening
        // a freshly-connected session's persisted buffers never blocks the
        // editor thread on the slow link (the connect worker already fetched
        // the bytes). A cold cache falls straight through to the live read.
        if let Some(content) = self.prewarm_content(path) {
            return Ok(content);
        }
        let path_str = path.to_string_lossy();
        let (data_chunks, _result) = self
            .channel
            .request_with_data_blocking("read", read_params(&path_str, None, None))
            .map_err(Self::to_io_error)?;

        // Collect all streaming data chunks
        let mut content = Vec::new();
        for chunk in data_chunks {
            if let Some(b64) = chunk.get("data").and_then(|v| v.as_str()) {
                let decoded = decode_base64(b64)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                content.extend(decoded);
            }
        }

        Ok(content)
    }

    fn read_range(&self, path: &Path, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        // Slice the prewarmed content when warm (same rationale as `read_file`).
        // Preserve the live path's short-read semantics: a range beyond the
        // cached bytes is an `UnexpectedEof`, not a silent truncation.
        if let Some(content) = self.prewarm_content(path) {
            let start = offset as usize;
            let end = start.saturating_add(len);
            if end <= content.len() {
                return Ok(content[start..end].to_vec());
            }
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "read_range: expected {} bytes at offset {}, cached file is {} bytes (path: {})",
                    len,
                    offset,
                    content.len(),
                    path.to_string_lossy(),
                ),
            ));
        }
        let path_str = path.to_string_lossy();
        let (data_chunks, result) = self
            .channel
            .request_with_data_blocking("read", read_params(&path_str, Some(offset), Some(len)))
            .map_err(Self::to_io_error)?;

        // Collect all streaming data chunks
        let mut content = Vec::new();
        for chunk in data_chunks {
            if let Some(b64) = chunk.get("data").and_then(|v| v.as_str()) {
                let decoded = decode_base64(b64)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                content.extend(decoded);
            }
        }

        // Get the size reported by the agent (how many bytes it actually read from the file)
        let agent_reported_size = result
            .get("size")
            .and_then(|v| v.as_u64())
            .map(|s| s as usize);

        // Validate that we received the expected number of bytes.
        // This matches LocalFileSystem::read_range which uses read_exact.
        // Short reads indicate file truncation, race conditions, or metadata mismatch.
        if content.len() != len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "read_range: expected {} bytes at offset {}, got {} (agent reported: {:?}, path: {})",
                    len,
                    offset,
                    content.len(),
                    agent_reported_size,
                    path_str
                ),
            ));
        }

        Ok(content)
    }

    fn count_line_feeds_in_range(&self, path: &Path, offset: u64, len: usize) -> io::Result<usize> {
        let path_str = path.to_string_lossy();
        let result = self
            .channel
            .request_blocking("count_lf", count_lf_params(&path_str, offset, len))
            .map_err(Self::to_io_error)?;

        result
            .get("count")
            .and_then(|v| v.as_u64())
            .map(|c| c as usize)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "missing count in count_lf response",
                )
            })
    }

    fn write_file(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        let path_str = path.to_string_lossy();
        self.channel
            .request_blocking("write", write_params(&path_str, data))
            .map_err(Self::to_io_error)?;
        Ok(())
    }

    fn create_file(&self, path: &Path) -> io::Result<Box<dyn FileWriter>> {
        // Create an empty file first
        self.write_file(path, &[])?;
        Ok(Box::new(RemoteFileWriter::new(
            self.channel.clone(),
            path.to_path_buf(),
        )))
    }

    fn open_file(&self, path: &Path) -> io::Result<Box<dyn FileReader>> {
        // Read the entire file into memory for seeking
        let data = self.read_file(path)?;
        Ok(Box::new(RemoteFileReader::new(data)))
    }

    fn open_file_for_write(&self, path: &Path) -> io::Result<Box<dyn FileWriter>> {
        Ok(Box::new(RemoteFileWriter::new(
            self.channel.clone(),
            path.to_path_buf(),
        )))
    }

    fn open_file_for_append(&self, path: &Path) -> io::Result<Box<dyn FileWriter>> {
        // Use append-only writer that sends only new data
        Ok(Box::new(AppendingRemoteFileWriter::new(
            self.channel.clone(),
            path.to_path_buf(),
        )))
    }

    fn set_file_length(&self, path: &Path, len: u64) -> io::Result<()> {
        let path_str = path.to_string_lossy();
        self.channel
            .request_blocking("truncate", truncate_params(&path_str, len))
            .map_err(Self::to_io_error)?;
        Ok(())
    }

    fn write_patched(&self, src_path: &Path, dst_path: &Path, ops: &[WriteOp]) -> io::Result<()> {
        // Convert WriteOps to protocol PatchOps
        let patch_ops: Vec<PatchOp> = ops
            .iter()
            .map(|op| match op {
                WriteOp::Copy { offset, len } => PatchOp::copy(*offset, *len),
                WriteOp::Insert { data } => PatchOp::insert(data),
            })
            .collect();

        let src_str = src_path.to_string_lossy();
        let dst_str = dst_path.to_string_lossy();
        let dst_param = if src_path == dst_path {
            None
        } else {
            Some(dst_str.as_ref())
        };

        self.channel
            .request_blocking("patch", patch_params(&src_str, dst_param, &patch_ops))
            .map_err(Self::to_io_error)?;
        Ok(())
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        let params = serde_json::json!({
            "from": from.to_string_lossy(),
            "to": to.to_string_lossy()
        });
        self.channel
            .request_blocking("mv", params)
            .map_err(Self::to_io_error)?;
        Ok(())
    }

    fn copy(&self, from: &Path, to: &Path) -> io::Result<u64> {
        let params = serde_json::json!({
            "from": from.to_string_lossy(),
            "to": to.to_string_lossy()
        });
        let result = self
            .channel
            .request_blocking("cp", params)
            .map_err(Self::to_io_error)?;

        Ok(result.get("size").and_then(|v| v.as_u64()).unwrap_or(0))
    }

    fn remove_file(&self, path: &Path) -> io::Result<()> {
        let params = serde_json::json!({"path": path.to_string_lossy()});
        self.channel
            .request_blocking("rm", params)
            .map_err(Self::to_io_error)?;
        Ok(())
    }

    fn remove_dir(&self, path: &Path) -> io::Result<()> {
        let params = serde_json::json!({"path": path.to_string_lossy()});
        self.channel
            .request_blocking("rmdir", params)
            .map_err(Self::to_io_error)?;
        Ok(())
    }

    fn metadata(&self, path: &Path) -> io::Result<FileMetadata> {
        if let Some(meta) = self.prewarm_meta(path) {
            return Ok(meta);
        }
        let path_str = path.to_string_lossy();
        let result = self
            .channel
            .request_blocking("stat", stat_params(&path_str, true))
            .map_err(Self::to_io_error)?;

        let rm: RemoteMetadata = serde_json::from_value(result)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        Ok(Self::convert_metadata(&rm, &name))
    }

    fn symlink_metadata(&self, path: &Path) -> io::Result<FileMetadata> {
        let path_str = path.to_string_lossy();
        let result = self
            .channel
            .request_blocking("stat", stat_params(&path_str, false))
            .map_err(Self::to_io_error)?;

        let rm: RemoteMetadata = serde_json::from_value(result)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        Ok(Self::convert_metadata(&rm, &name))
    }

    fn is_dir(&self, path: &Path) -> io::Result<bool> {
        let path_str = path.to_string_lossy();
        let result = self
            .channel
            .request_blocking("stat", stat_params(&path_str, true))
            .map_err(Self::to_io_error)?;

        Ok(result.get("dir").and_then(|v| v.as_bool()).unwrap_or(false))
    }

    fn is_file(&self, path: &Path) -> io::Result<bool> {
        let path_str = path.to_string_lossy();
        let result = self
            .channel
            .request_blocking("stat", stat_params(&path_str, true))
            .map_err(Self::to_io_error)?;

        Ok(result
            .get("file")
            .and_then(|v| v.as_bool())
            .unwrap_or(false))
    }

    fn set_permissions(&self, path: &Path, permissions: &FilePermissions) -> io::Result<()> {
        #[cfg(unix)]
        {
            let params = serde_json::json!({
                "path": path.to_string_lossy(),
                "mode": permissions.mode()
            });
            self.channel
                .request_blocking("chmod", params)
                .map_err(Self::to_io_error)?;
        }
        #[cfg(not(unix))]
        {
            let _ = (path, permissions);
        }
        Ok(())
    }

    fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        let path_str = path.to_string_lossy();
        let result = self
            .channel
            .request_blocking("ls", ls_params(&path_str))
            .map_err(Self::to_io_error)?;

        let entries: Vec<RemoteDirEntry> = result
            .get("entries")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        Ok(entries.iter().map(Self::convert_dir_entry).collect())
    }

    fn create_dir(&self, path: &Path) -> io::Result<()> {
        let params = serde_json::json!({"path": path.to_string_lossy()});
        self.channel
            .request_blocking("mkdir", params)
            .map_err(Self::to_io_error)?;
        Ok(())
    }

    fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        let params = serde_json::json!({
            "path": path.to_string_lossy(),
            "parents": true
        });
        self.channel
            .request_blocking("mkdir", params)
            .map_err(Self::to_io_error)?;
        Ok(())
    }

    fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        if let Some(canon) = self.prewarm_canon(path) {
            return Ok(canon);
        }
        let params = serde_json::json!({"path": path.to_string_lossy()});
        let result = self
            .channel
            .request_blocking("realpath", params)
            .map_err(Self::to_io_error)?;

        let canonical = result.get("path").and_then(|v| v.as_str()).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "missing path in response")
        })?;

        Ok(PathBuf::from(canonical))
    }

    fn clear_prewarm(&self) {
        let mut cache = self.prewarm.lock().unwrap();
        *cache = PrewarmCache::default();
    }

    fn current_uid(&self) -> u32 {
        // We don't know the remote user's UID easily, return 0
        // This is used for ownership checks which we skip for remote
        0
    }

    fn remote_connection_info(&self) -> Option<&str> {
        Some(&self.connection_string)
    }

    fn is_remote_connected(&self) -> bool {
        self.channel.is_connected()
    }

    fn remote_channel_id(&self) -> Option<u64> {
        Some(self.channel.id())
    }

    fn remote_reconnect_notify(&self) -> Option<std::sync::Arc<tokio::sync::Notify>> {
        Some(self.channel.reconnect_notify())
    }

    fn home_dir(&self) -> io::Result<PathBuf> {
        // Served from the connect-time cache on the hot path (workspace
        // restore / file open / file explorer), so the editor thread doesn't
        // block; see `sys_info`. A remote that couldn't answer `info` never
        // gets this far — the connect-time liveness gate (`prime_sys_info`)
        // rejects it before the session is promoted.
        self.sys_info()
            .and_then(|info| info.home)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "remote home directory unknown"))
    }

    fn unique_temp_path(&self, dest_path: &Path) -> PathBuf {
        // Use the remote system's temp directory instead of hardcoding /tmp,
        // which doesn't exist on Windows remotes. Served from the connect-time
        // cache when primed (see `sys_info`); falls back to /tmp if the info
        // request fails (e.g. older agent without temp_dir support).
        let temp_dir = self
            .sys_info()
            .map(|i| i.temp_dir)
            .unwrap_or_else(|| PathBuf::from("/tmp"));
        let file_name = dest_path
            .file_name()
            .unwrap_or_else(|| std::ffi::OsStr::new("fresh-save"));
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        temp_dir.join(format!(
            "{}-{}-{}.tmp",
            file_name.to_string_lossy(),
            std::process::id(),
            timestamp
        ))
    }

    fn search_file(
        &self,
        path: &Path,
        pattern: &str,
        opts: &crate::model::filesystem::FileSearchOptions,
        cursor: &mut crate::model::filesystem::FileSearchCursor,
    ) -> io::Result<Vec<crate::model::filesystem::SearchMatch>> {
        if cursor.done {
            return Ok(vec![]);
        }

        let path_str = path.to_string_lossy();
        let mut params = serde_json::json!({
            "path": path_str,
            "pattern": pattern,
            "fixed_string": opts.fixed_string,
            "case_sensitive": opts.case_sensitive,
            "whole_word": opts.whole_word,
            "max_matches": opts.max_matches,
            "offset": cursor.offset,
            "running_line": cursor.running_line,
        });
        if let Some(end) = cursor.end_offset {
            params["end_offset"] = serde_json::json!(end);
        }

        let result = self
            .channel
            .request_blocking("search_file", params)
            .map_err(Self::to_io_error)?;

        cursor.offset = result
            .get("next_offset")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        cursor.running_line = result
            .get("running_line")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;
        cursor.done = result.get("done").and_then(|v| v.as_bool()).unwrap_or(true);

        let matches: Vec<crate::model::filesystem::SearchMatch> = result
            .get("matches")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        Some(crate::model::filesystem::SearchMatch {
                            byte_offset: m.get("byte_offset")?.as_u64()? as usize,
                            length: m.get("length")?.as_u64()? as usize,
                            line: m.get("line")?.as_u64()? as usize,
                            column: m.get("column")?.as_u64()? as usize,
                            context: m.get("context")?.as_str()?.to_string(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(matches)
    }

    fn sudo_write(
        &self,
        path: &Path,
        data: &[u8],
        mode: u32,
        uid: u32,
        gid: u32,
    ) -> io::Result<()> {
        let path_str = path.to_string_lossy();
        self.channel
            .request_blocking(
                "sudo_write",
                sudo_write_params(&path_str, data, mode, uid, gid),
            )
            .map_err(Self::to_io_error)?;
        Ok(())
    }

    fn walk_files(
        &self,
        root: &Path,
        skip_dirs: &[&str],
        cancel: &std::sync::atomic::AtomicBool,
        on_file: &mut dyn FnMut(&Path, &str) -> bool,
    ) -> io::Result<()> {
        let path_str = root.to_string_lossy();
        let params = serde_json::json!({
            "path": path_str,
            "skip_dirs": skip_dirs,
        });

        // Server-side walk: the remote agent walks the tree and streams
        // back batches of relative paths.  We process each batch as it
        // arrives, keeping memory bounded.
        let (mut data_rx, result_rx) = self
            .channel
            .request_streaming_blocking("walk_files", params)
            .map_err(Self::to_io_error)?;

        // Process streaming batches
        while let Some(data) = data_rx.blocking_recv() {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                // Drop receivers — server sees send fail and stops
                drop(data_rx);
                drop(result_rx);
                return Ok(());
            }

            if let Some(files) = data.get("files").and_then(|v| v.as_array()) {
                for file in files {
                    if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                        drop(result_rx);
                        return Ok(());
                    }
                    if let Some(rel) = file.as_str() {
                        let abs = root.join(rel);
                        if !on_file(&abs, rel) {
                            // Caller limit reached — drop receivers to signal
                            // cancellation to the server
                            drop(result_rx);
                            return Ok(());
                        }
                    }
                }
            }
        }

        // Drain the final result — channel may already be closed if the
        // server finished before we read this, which is fine.
        drop(result_rx.blocking_recv());
        Ok(())
    }
}

/// Remote file reader - wraps in-memory data
struct RemoteFileReader {
    cursor: Cursor<Vec<u8>>,
}

impl RemoteFileReader {
    fn new(data: Vec<u8>) -> Self {
        Self {
            cursor: Cursor::new(data),
        }
    }
}

impl Read for RemoteFileReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.cursor.read(buf)
    }
}

impl Seek for RemoteFileReader {
    fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
        self.cursor.seek(pos)
    }
}

impl FileReader for RemoteFileReader {}

/// Remote file writer - buffers writes and flushes on sync
struct RemoteFileWriter {
    channel: Arc<AgentChannel>,
    path: PathBuf,
    buffer: Vec<u8>,
}

impl RemoteFileWriter {
    fn new(channel: Arc<AgentChannel>, path: PathBuf) -> Self {
        Self {
            channel,
            path,
            buffer: Vec::new(),
        }
    }
}

impl Write for RemoteFileWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        // Flush is a no-op; actual write happens on sync_all
        Ok(())
    }
}

impl FileWriter for RemoteFileWriter {
    fn sync_all(&self) -> io::Result<()> {
        let path_str = self.path.to_string_lossy();
        self.channel
            .request_blocking("write", write_params(&path_str, &self.buffer))
            .map_err(RemoteFileSystem::to_io_error)?;
        Ok(())
    }
}

/// Remote file writer for append operations - only sends new data
struct AppendingRemoteFileWriter {
    channel: Arc<AgentChannel>,
    path: PathBuf,
    buffer: Vec<u8>,
}

impl AppendingRemoteFileWriter {
    fn new(channel: Arc<AgentChannel>, path: PathBuf) -> Self {
        Self {
            channel,
            path,
            buffer: Vec::new(),
        }
    }
}

impl Write for AppendingRemoteFileWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl FileWriter for AppendingRemoteFileWriter {
    fn sync_all(&self) -> io::Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let path_str = self.path.to_string_lossy();
        self.channel
            .request_blocking("append", append_params(&path_str, &self.buffer))
            .map_err(RemoteFileSystem::to_io_error)?;
        Ok(())
    }
}

#[cfg(test)]
impl RemoteFileSystem {
    /// Seed the prewarm content cache directly, so the read-side consultation
    /// can be unit-tested without standing up a live agent.
    fn test_seed_prewarm_content(&self, path: PathBuf, content: Vec<u8>) {
        self.prewarm.lock().unwrap().content.insert(path, content);
    }

    /// True when every prewarm cache is empty (used to assert `clear_prewarm`).
    fn test_prewarm_is_empty(&self) -> bool {
        let cache = self.prewarm.lock().unwrap();
        cache.content.is_empty() && cache.meta.is_empty() && cache.canon.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `RemoteFileSystem` over a dead in-memory transport. The agent
    /// channel's tasks start (a runtime must be current) but nothing answers —
    /// which is exactly the point: a prewarmed read must be served from cache
    /// without ever touching the link.
    fn fs_over_dead_transport() -> (tokio::runtime::Runtime, RemoteFileSystem) {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let fs = rt.block_on(async {
            let (near, _far) = tokio::io::duplex(64);
            let (reader, writer) = tokio::io::split(near);
            let channel = std::sync::Arc::new(AgentChannel::from_transport(
                tokio::io::BufReader::new(reader),
                writer,
                8,
            ));
            RemoteFileSystem::new(channel, "test".to_string())
        });
        (rt, fs)
    }

    #[test]
    fn prewarmed_reads_are_served_from_cache() {
        let (_rt, fs) = fs_over_dead_transport();
        let path = PathBuf::from("/proj/notes.txt");
        fs.test_seed_prewarm_content(path.clone(), b"REMOTE NOTES\n".to_vec());

        // Full read and in-range slice come from the cache — no round-trip, so
        // this returns instantly even though the transport is dead.
        assert_eq!(fs.read_file(&path).unwrap(), b"REMOTE NOTES\n");
        assert_eq!(fs.read_range(&path, 0, 6).unwrap(), b"REMOTE");
        assert!(fs.open_file(&path).is_ok());

        // A range past the cached content preserves the live path's short-read
        // semantics rather than truncating.
        assert_eq!(
            fs.read_range(&path, 7, 100).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );

        // Once cleared the cache is empty; later reads go back to the link.
        fs.clear_prewarm();
        assert!(fs.test_prewarm_is_empty());
    }

    #[test]
    fn test_convert_metadata() {
        // Use the current user's uid/gid so the file appears writable regardless
        // of which user runs the test (on Unix, is_readonly checks effective user).
        #[cfg(unix)]
        let (uid, gid) = {
            let (euid, groups) = crate::model::filesystem::StdFileSystem::current_user_groups();
            (euid, *groups.first().unwrap_or(&0u32))
        };
        #[cfg(not(unix))]
        let (uid, gid) = (1000u32, 1000u32);

        let rm = RemoteMetadata {
            size: 1234,
            mtime: 1700000000,
            mode: 0o644,
            uid,
            gid,
            dir: false,
            file: true,
            link: false,
        };

        let meta = RemoteFileSystem::convert_metadata(&rm, "test.txt");
        assert_eq!(meta.size, 1234);
        assert!(!meta.is_hidden);
        assert!(!meta.is_readonly);

        let meta = RemoteFileSystem::convert_metadata(&rm, ".hidden");
        assert!(meta.is_hidden);
    }

    #[test]
    fn test_convert_dir_entry() {
        let re = RemoteDirEntry {
            name: "file.rs".to_string(),
            path: "/home/user/file.rs".to_string(),
            dir: false,
            file: true,
            link: false,
            link_dir: false,
            size: 100,
            mtime: 1700000000,
            mode: 0o644,
        };

        let entry = RemoteFileSystem::convert_dir_entry(&re);
        assert_eq!(entry.name, "file.rs");
        assert_eq!(entry.entry_type, EntryType::File);
        assert!(!entry.is_symlink());
    }
}
