//! Process and filesystem helpers that differ between Unix and Windows.

use std::process::{Command, Stdio};

/// Convert a Windows verbatim path (the form returned by
/// [`std::fs::canonicalize`]) to the regular path spelling understood by
/// command-line tools such as Git. Windows APIs accept both spellings, but Git
/// treats a `\\?\` path supplied through `GIT_DIR`/`GIT_WORK_TREE` as a
/// different repository.
pub fn external_path(path: &std::path::Path) -> std::path::PathBuf {
    #[cfg(windows)]
    {
        let value = path.to_string_lossy();
        if let Some(unc) = value.strip_prefix(r"\\?\UNC\") {
            return std::path::PathBuf::from(format!(r"\\{unc}"));
        }
        if let Some(normal) = value.strip_prefix(r"\\?\") {
            return std::path::PathBuf::from(normal);
        }
    }
    path.to_path_buf()
}

/// Run a Windows batch shim (`.cmd`/`.bat`) through `cmd.exe`. Node-based CLI
/// installers commonly put only these shims on PATH; `CreateProcess` cannot
/// execute them directly.
pub fn command(program: impl AsRef<std::ffi::OsStr>) -> Command {
    let program = program.as_ref();
    #[cfg(windows)]
    if std::path::Path::new(program)
        .extension()
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("cmd") || extension.eq_ignore_ascii_case("bat")
        })
    {
        let mut command = Command::new("cmd.exe");
        command.args(["/D", "/S", "/C"]).arg(program);
        return command;
    }
    Command::new(program)
}

/// Tokio counterpart to [`command`].
pub fn tokio_command(program: impl AsRef<std::ffi::OsStr>) -> tokio::process::Command {
    let program = program.as_ref();
    #[cfg(windows)]
    if std::path::Path::new(program)
        .extension()
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("cmd") || extension.eq_ignore_ascii_case("bat")
        })
    {
        let mut command = tokio::process::Command::new("cmd.exe");
        command.args(["/D", "/S", "/C"]).arg(program);
        return command;
    }
    tokio::process::Command::new(program)
}

/// Put a foreground child in a private process group so a timeout/cancel can
/// terminate the child tree without affecting the dashboard process.
pub fn new_process_group(cmd: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        cmd.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt as _;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
    }
}

/// Tokio counterpart to [`new_process_group`].
pub fn new_process_group_tokio(cmd: &mut tokio::process::Command) {
    #[cfg(unix)]
    {
        cmd.process_group(0);
    }
    #[cfg(windows)]
    {
        cmd.creation_flags(0x0000_0200 | 0x0800_0000);
    }
}

/// Detach `cmd` from the current session so it outlives this process.
///
/// Unix: new process group. Windows: new process group, detached, no console.
pub fn detach(cmd: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        cmd.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt as _;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
    }
}

/// Whether `pid` still names a live (non-zombie) process.
pub fn pid_alive(pid: &str) -> bool {
    let pid = pid.trim();
    if pid.is_empty() {
        return false;
    }
    #[cfg(unix)]
    {
        match Command::new("ps")
            .args(["-o", "stat=", "-p", pid])
            .stderr(Stdio::null())
            .output()
        {
            Ok(output) if output.status.success() => {
                let stat = String::from_utf8_lossy(&output.stdout);
                let stat = stat.trim();
                !stat.is_empty() && !stat.starts_with('Z')
            }
            _ => false,
        }
    }
    #[cfg(windows)]
    {
        match Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
            .stderr(Stdio::null())
            .output()
        {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stdout = stdout.trim();
                !stdout.is_empty() && !stdout.contains("No tasks") && stdout.contains(pid)
            }
            _ => false,
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
        false
    }
}

/// Terminate `pid` and its descendants. Returns whether a kill was issued.
pub fn kill_tree(pid: &str) -> bool {
    let pid = pid.trim();
    if pid.is_empty() {
        return false;
    }
    #[cfg(unix)]
    {
        let group = Command::new("kill")
            .args(["-TERM", "--", &format!("-{pid}")])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false);
        if group {
            return true;
        }
        Command::new("kill")
            .args(["-TERM", pid])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }
    #[cfg(windows)]
    {
        Command::new("taskkill")
            .args(["/PID", pid, "/T", "/F"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
        false
    }
}

/// `rename` failed because the paths are on different volumes.
pub fn is_exdev(error: &std::io::Error) -> bool {
    #[cfg(unix)]
    {
        error.raw_os_error() == Some(libc::EXDEV)
    }
    #[cfg(windows)]
    {
        // ERROR_NOT_SAME_DEVICE
        error.raw_os_error() == Some(17)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = error;
        false
    }
}

/// Single-quote a string for PowerShell.
#[cfg(windows)]
pub fn ps_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}
