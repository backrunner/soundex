//! Benchmark artifact, platform, statistics and environment helpers.

use super::*;

pub(super) fn validate_artifact(path: &Path) -> Result<ArtifactEvidence, Box<dyn Error>> {
    let canonical = path.canonicalize().map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("cannot resolve benchmark model {}: {error}", path.display()),
        )
    })?;
    let metadata = canonical.metadata()?;
    if !metadata.is_file() {
        return Err("SOUNDEX_BENCH_MODEL must name a regular ONNX file".into());
    }
    let sha256 = sha256_file(&canonical)?;
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/identity.onnx");
    let fixture_hash = fixture
        .is_file()
        .then(|| sha256_file(&fixture))
        .transpose()?;
    if fixture_hash.as_deref() == Some(&sha256)
        || canonical.file_name().and_then(|name| name.to_str()) == Some("identity.onnx")
    {
        return Err(
            "identity.onnx is a correctness fixture, not a production benchmark model".into(),
        );
    }
    if metadata.len() < MIN_PRODUCTION_MODEL_BYTES {
        return Err(format!(
            "model is only {} bytes; production benchmark requires a non-trivial artifact of at least {MIN_PRODUCTION_MODEL_BYTES} bytes",
            metadata.len()
        )
        .into());
    }
    if metadata.len() >= MAX_PRODUCTION_MODEL_BYTES {
        return Err(format!(
            "model is {} bytes; production benchmark requires the FP32 artifact to be smaller than {MAX_PRODUCTION_MODEL_BYTES} bytes",
            metadata.len()
        )
        .into());
    }
    Ok(ArtifactEvidence {
        path: canonical.display().to_string(),
        sha256,
        size_bytes: metadata.len(),
    })
}

pub(super) fn forced_enhancement_input(sample_rate: u32, channels: u16) -> Vec<f32> {
    (0..HOP_SIZE)
        .flat_map(|sample| {
            let time = sample as f32 / sample_rate as f32;
            let left = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * time).sin();
            let right = 0.5 * (2.0 * std::f32::consts::PI * 880.0 * time).sin();
            [left, right].into_iter().take(channels as usize)
        })
        .collect()
}

pub(super) fn latency_summary(samples_ns: &[u64]) -> LatencySummary {
    let mut sorted = samples_ns.to_vec();
    sorted.sort_unstable();
    LatencySummary {
        p50_us: percentile_ns(&sorted, 0.50) / 1_000.0,
        p95_us: percentile_ns(&sorted, 0.95) / 1_000.0,
        p99_us: percentile_ns(&sorted, 0.99) / 1_000.0,
        max_us: sorted.last().copied().unwrap_or(0) as f64 / 1_000.0,
    }
}

pub(super) fn percentile_ns(sorted: &[u64], percentile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (percentile * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)] as f64
}

pub(super) fn sha256_file(path: &Path) -> Result<String, Box<dyn Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub(super) fn write_report(path: &Path, report: &BenchmarkReport) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut writer, report)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

pub(super) fn required_path(name: &str) -> Result<PathBuf, Box<dyn Error>> {
    env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .map(workspace_relative)
        .ok_or_else(|| {
            format!("{name} is required and must name the production ONNX artifact").into()
        })
}

pub(super) fn workspace_relative(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        workspace_root().join(path)
    }
}

pub(super) fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub(super) fn env_number<T>(name: &str, default: T) -> Result<T, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: Error + 'static,
{
    match env::var(name) {
        Ok(value) => value
            .parse::<T>()
            .map_err(|error| format!("invalid {name}={value:?}: {error}").into()),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

pub(super) fn env_flag(name: &str) -> bool {
    matches!(env::var(name).as_deref(), Ok("1" | "true" | "yes"))
}

pub(super) fn hardware_evidence() -> HardwareEvidence {
    let cpu = cpu_name();
    HardwareEvidence {
        label: env::var("SOUNDEX_BENCH_HARDWARE_LABEL").unwrap_or_else(|_| cpu.clone()),
        cpu,
        architecture: env::consts::ARCH.to_owned(),
        operating_system: env::consts::OS.to_owned(),
        host: env::var("HOSTNAME")
            .or_else(|_| env::var("COMPUTERNAME"))
            .unwrap_or_else(|_| "unknown".to_owned()),
        power_mode: env::var("SOUNDEX_BENCH_POWER_MODE")
            .unwrap_or_else(|_| "not-recorded".to_owned()),
        rustc: command_output("rustc", &["--version"]).unwrap_or_else(|| "unknown".to_owned()),
    }
}

pub(super) fn cpu_name() -> String {
    if cfg!(target_os = "macos") {
        return command_output("/usr/sbin/sysctl", &["-n", "machdep.cpu.brand_string"])
            .unwrap_or_else(|| "unknown Apple CPU".to_owned());
    }
    if cfg!(target_os = "linux") {
        if let Ok(contents) = fs::read_to_string("/proc/cpuinfo") {
            if let Some(value) = contents.lines().find_map(|line| {
                let (key, value) = line.split_once(':')?;
                matches!(key.trim(), "model name" | "Processor").then(|| value.trim().to_owned())
            }) {
                return value;
            }
        }
    }
    env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unknown".to_owned())
}

pub(super) fn command_output(program: &str, arguments: &[&str]) -> Option<String> {
    let output = Command::new(program).args(arguments).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

#[cfg(target_os = "linux")]
pub(super) fn resident_set_size() -> (Option<u64>, &'static str) {
    let value = fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|contents| {
            contents.lines().find_map(|line| {
                let value = line.strip_prefix("VmHWM:")?;
                value.split_whitespace().next()?.parse::<u64>().ok()
            })
        })
        .map(|kilobytes| kilobytes * 1024);
    (value, "linux-vmhwm")
}

#[cfg(target_os = "macos")]
pub(super) fn resident_set_size() -> (Option<u64>, &'static str) {
    let mut usage = MaybeUninit::<libc::rusage>::zeroed();
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    let value = (result == 0).then(|| unsafe { usage.assume_init().ru_maxrss as u64 });
    (value, "macos-getrusage-ru_maxrss")
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) fn resident_set_size() -> (Option<u64>, &'static str) {
    let process_id = process::id().to_string();
    let value = command_output("ps", &["-o", "rss=", "-p", &process_id])
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map(|kilobytes| kilobytes * 1024);
    (value, "ps-current-rss")
}

pub(super) fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}
