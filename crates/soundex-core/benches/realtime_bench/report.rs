//! Stream raw trace arrays to disk without retaining four JSON value trees.

use std::{
    fs::{self, File, OpenOptions},
    io::{self, BufReader, BufWriter, Write},
    path::PathBuf,
};

use serde::Serialize;
use serde_json::Value;

#[derive(Serialize)]
pub(super) struct Case {
    #[serde(flatten)]
    pub metadata: Value,
    pub callback_samples_ns: Vec<u64>,
    pub worker_presentation_missed: Vec<bool>,
}

struct Temporary(PathBuf);

impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

pub(super) struct Report {
    target: PathBuf,
    cases: Vec<Temporary>,
    serial: u64,
}

impl Report {
    pub fn new(target: PathBuf) -> Self {
        Self {
            target,
            cases: Vec::new(),
            serial: 0,
        }
    }

    fn temporary(&mut self) -> io::Result<(Temporary, File)> {
        self.serial += 1;
        let name = self.target.file_name().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "report needs a filename")
        })?;
        let path = self.target.with_file_name(format!(
            ".{}.{}.{}.tmp",
            name.to_string_lossy(),
            std::process::id(),
            self.serial
        ));
        // Never follow/overwrite a pre-existing temporary file.
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        Ok((Temporary(path), file))
    }

    pub fn append(&mut self, header: &Value, case: &Case) -> io::Result<()> {
        if !header.is_object()
            || header.as_object().is_some_and(|object| object.is_empty())
            || header.get("cases").is_some()
            || !case.metadata.is_object()
            || case.metadata.get("callback_samples_ns").is_some()
            || case.metadata.get("worker_presentation_missed").is_some()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid report metadata",
            ));
        }
        let (temporary, file) = self.temporary()?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, case)?;
        writer.flush()?;
        self.cases.push(temporary);
        let (published, file) = self.temporary()?;
        let mut writer = BufWriter::new(file);
        let mut prefix = serde_json::to_vec_pretty(header)?;
        // An object serialized by serde_json ends with this delimiter.
        if prefix.pop() != Some(b'}') {
            return Err(io::Error::other("invalid header serialization"));
        }
        writer.write_all(&prefix)?;
        writer.write_all(b",\n\"cases\": [\n")?;
        for (index, case) in self.cases.iter().enumerate() {
            if index != 0 {
                writer.write_all(b",\n")?;
            }
            io::copy(&mut BufReader::new(File::open(&case.0)?), &mut writer)?;
        }
        writer.write_all(b"\n]}\n")?;
        writer.flush()?;
        drop(writer);
        // Preserve the previously completed cases if publication fails.
        fs::rename(&published.0, &self.target)
    }
}
