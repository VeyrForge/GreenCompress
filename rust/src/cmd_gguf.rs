//! CLI wrappers for GGUF export and Green model pack (Python scripts).

use std::env;
use std::path::PathBuf;
use std::process::Command;

use crate::error::{fail, Result};
use crate::types::Args;
use crate::util::{get_bool, get_optional_string, get_string};

pub fn cmd_export_gguf(args: &Args) -> Result<()> {
    let gguf = get_string(args, "gguf", "")?;
    let out = get_string(args, "out", "")?;
    let method = get_optional_string(args, "method", "green_optimal");
    let verify = get_bool(args, "verify", false);
    run_python_script("export_gguf.py", &[
        ("--gguf", gguf.as_str()),
        ("--out", out.as_str()),
        ("--method", method.as_str()),
    ], verify)
}

pub fn cmd_pack_model(args: &Args) -> Result<()> {
    let gguf = get_string(args, "gguf", "")?;
    let out = get_string(args, "out", "")?;
    let method = get_optional_string(args, "method", "green_optimal");
    let verify = get_bool(args, "verify", false);
    run_python_script("pack_model.py", &[
        ("--gguf", gguf.as_str()),
        ("--out", out.as_str()),
        ("--method", method.as_str()),
    ], verify)
}

fn run_python_script(script: &str, flags: &[(&str, &str)], verify: bool) -> Result<()> {
    let root = repo_root();
    let script_path = root.join("scripts").join(script);
    if !script_path.is_file() {
        return Err(fail(format!(
            "script not found: {} (set GREENCOMPRESS_ROOT or run from repo)",
            script_path.display()
        )));
    }

    let python = env::var("GREEN_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let mut cmd = Command::new(&python);
    cmd.arg(&script_path);
    for (k, v) in flags {
        cmd.arg(k).arg(v);
    }
    if verify {
        cmd.arg("--verify");
    }

    let status = cmd
        .status()
        .map_err(|e| fail(format!("failed to run {python} {script}: {e}")))?;
    if !status.success() {
        return Err(fail(format!("{script} exited with {status}")));
    }
    Ok(())
}

fn repo_root() -> PathBuf {
    if let Ok(p) = env::var("GREENCOMPRESS_ROOT") {
        return PathBuf::from(p);
    }
    if let Ok(exe) = env::current_exe() {
        let mut dir = exe.parent().map(PathBuf::from);
        while let Some(ref path) = dir {
            if path.join("scripts").join("export_gguf.py").is_file() {
                return path.clone();
            }
            dir = path.parent().map(PathBuf::from);
        }
    }
    if let Ok(cwd) = env::current_dir() {
        let mut dir = Some(cwd);
        while let Some(ref path) = dir {
            if path.join("scripts").join("export_gguf.py").is_file() {
                return path.clone();
            }
            dir = path.parent().map(PathBuf::from);
        }
    }
    PathBuf::from(".")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_root_has_scripts() {
        let root = repo_root();
        assert!(
            root.join("scripts").join("export_gguf.py").is_file(),
            "expected scripts/export_gguf.py under {}",
            root.display()
        );
    }
}
