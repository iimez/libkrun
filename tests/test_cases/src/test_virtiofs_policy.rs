#![cfg(all(
    any(feature = "host", target_os = "linux"),
    not(feature = "dynamic-linking")
))]

//! A virtiofs policy set with `FsDevice::set_policy` refuses what the
//! guest does under a protected directory while the host says so, and lets it
//! through from the next request once the host changes its mind, without
//! restarting the guest. Symlinks, hard links and directories opened earlier
//! don't get around it, and the host finds the directory unchanged.

use macros::{guest, host};

pub struct TestVirtiofsPolicy;

/// The guest connects out on this port to synchronize with the host.
const CONTROL_PORT: u32 = 1234;
const TAG: &str = "shared";
const EACCES: i32 = 13;

#[host]
mod host {
    use super::*;

    use std::fs;
    use std::io::{Read, Write};
    use std::os::unix::fs::MetadataExt;
    use std::os::unix::net::UnixListener;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    use crate::common::{build_init_config, setup_standard_devices};
    use crate::{Test, TestOutcome, TestSetup};

    /// The host protects the directory twice; `check` compares what was in it when each
    /// period began and when it ended.
    const PERIODS: [&str; 2] = ["first", "second"];

    struct ProtectDirectory {
        locked: Arc<AtomicBool>,
        /// The device and inode of `protected/linked`, which is also `free/alias`.
        linked: (u64, u64),
    }

    impl ProtectDirectory {
        fn protects(&self, object: &krun::FsObject) -> bool {
            object
                .path()
                .is_some_and(|path| path.starts_with("protected"))
                || object
                    .node()
                    .is_some_and(|node| (node.dev, node.ino) == self.linked)
        }
    }

    impl krun::FsPolicy for ProtectDirectory {
        fn check(&self, request: &krun::FsRequest) -> krun::FsVerdict {
            let flags = request.flags();
            let changes = match request.op() {
                krun::FsOp::Open => flags & 3 != 0 || flags & 0o1000 != 0,
                krun::FsOp::Opendir
                | krun::FsOp::Readlink
                | krun::FsOp::Getxattr
                | krun::FsOp::Listxattr => false,
                _ => true,
            };
            let inside = self.protects(request.object())
                || request.target().is_some_and(|target| self.protects(target));
            if changes && inside && self.locked.load(Ordering::SeqCst) {
                return krun::FsVerdict::Deny(EACCES);
            }
            krun::FsVerdict::Allow
        }
    }

    /// What the guest could change in `dir`.
    fn snapshot(dir: &Path) -> String {
        let mut paths: Vec<PathBuf> = fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        paths.sort();
        let mut out = String::new();
        for path in paths {
            let meta = fs::symlink_metadata(&path).unwrap();
            out += &format!(
                "{path:?} mode {:o} ino {} nlink {} mtime {}.{:09} ctime {}.{:09}\n",
                meta.mode(),
                meta.ino(),
                meta.nlink(),
                meta.mtime(),
                meta.mtime_nsec(),
                meta.ctime(),
                meta.ctime_nsec()
            );
            if meta.is_dir() {
                out += &snapshot(&path);
            } else if meta.is_file() {
                out += &format!("{:?}\n", fs::read_to_string(&path).unwrap());
            }
        }
        out
    }

    fn record(test_dir: &Path, period: &str, edge: &str) {
        let snapshot = snapshot(&test_dir.join("shared/protected"));
        fs::write(test_dir.join(format!("{period}.{edge}")), snapshot).unwrap();
    }

    fn run(listener: UnixListener, locked: Arc<AtomicBool>, test_dir: PathBuf) {
        let (mut control, _) = listener.accept().unwrap();
        // The guest has been refused.
        control.read_exact(&mut [0]).unwrap();
        record(&test_dir, PERIODS[0], "end");
        locked.store(false, Ordering::SeqCst);
        control.write_all(&[1]).unwrap();
        // The guest has opened the protected directory.
        control.read_exact(&mut [0]).unwrap();
        locked.store(true, Ordering::SeqCst);
        record(&test_dir, PERIODS[1], "start");
        control.write_all(&[1]).unwrap();
        // The guest has been refused through the directory it opened.
        control.read_exact(&mut [0]).unwrap();
        record(&test_dir, PERIODS[1], "end");
        control.write_all(&[1]).unwrap();
        // Keep the connection open until the guest is done.
        let _ = control.read(&mut [0]);
    }

    impl Test for TestVirtiofsPolicy {
        fn timeout_secs(&self) -> u64 {
            30
        }

        fn start_vm(self: Box<Self>, test_setup: TestSetup) -> anyhow::Result<()> {
            let shared = test_setup.tmp_dir.join("shared");
            fs::create_dir_all(shared.join("protected"))?;
            fs::create_dir_all(shared.join("free"))?;
            fs::write(shared.join("protected/file"), "kept")?;
            fs::write(shared.join("protected/linked"), "linked")?;
            fs::hard_link(shared.join("protected/linked"), shared.join("free/alias"))?;
            let linked = fs::metadata(shared.join("protected/linked"))?;
            record(&test_setup.tmp_dir, PERIODS[0], "start");

            let locked = Arc::new(AtomicBool::new(true));
            let policy = ProtectDirectory {
                locked: locked.clone(),
                linked: (linked.dev(), linked.ino()),
            };
            let control_sock = test_setup.tmp_dir.join("control.sock");
            let listener = UnixListener::bind(&control_sock)?;
            let test_dir = test_setup.tmp_dir.clone();
            thread::spawn(move || run(listener, locked, test_dir));

            let init_config = build_init_config(&test_setup.test_case, &[]);
            let stdin = std::io::stdin();
            let stdout = std::io::stdout();
            let stderr = std::io::stderr();
            let (mut devices, payload) =
                setup_standard_devices(&test_setup, &init_config, &stdin, &stdout, &stderr)?;

            let mut fs_dev = krun::FsDevice::new(TAG, shared.to_str().unwrap())
                .map_err(|e| anyhow::anyhow!("FsDevice::new: {e:?}"))?;
            fs_dev
                .set_policy(Arc::new(policy))
                .map_err(|e| anyhow::anyhow!("set_policy: {e:?}"))?;
            devices.add(fs_dev);

            let mut vsock = krun::VsockDevice::new(3, krun::TsiFlags::empty())
                .map_err(|e| anyhow::anyhow!("VsockDevice: {e:?}"))?;
            vsock.add_unix_port(CONTROL_PORT, control_sock.to_str().unwrap(), false);
            devices.add(vsock);

            let vmm = krun::VmmBuilder::new()
                .vcpus(1)
                .map_err(|e| anyhow::anyhow!("vcpus: {e:?}"))?
                .ram_mib(512)
                .map_err(|e| anyhow::anyhow!("ram_mib: {e:?}"))?
                .payload(payload)
                .devices(devices)
                .build()
                .map_err(|e| anyhow::anyhow!("build: {e:?}"))?;

            vmm.run();
            unreachable!()
        }

        fn check(self: Box<Self>, stdout: Vec<u8>, test_setup: TestSetup) -> TestOutcome {
            let output = String::from_utf8(stdout).unwrap();
            if output != "OK\n" {
                return TestOutcome::Fail(format!(
                    "expected exactly {:?}, got {:?}",
                    "OK\n", output
                ));
            }
            for period in PERIODS {
                let path = |edge| test_setup.tmp_dir.join(format!("{period}.{edge}"));
                let start = fs::read_to_string(path("start")).unwrap();
                let end = fs::read_to_string(path("end")).unwrap();
                if start != end {
                    return TestOutcome::Fail(format!(
                        "the guest changed the protected directory the {period} time it was \
                         protected, from\n{start}to\n{end}"
                    ));
                }
            }
            TestOutcome::Pass
        }
    }
}

#[guest]
mod guest {
    use super::*;
    use crate::Test;
    use nix::libc::{self, VMADDR_CID_HOST};
    use nix::sys::socket::{AddressFamily, SockFlag, SockType, VsockAddr, connect, socket};
    use std::ffi::CString;
    use std::fs;
    use std::io::{Read, Write};
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::fs::symlink;
    use std::os::unix::net::UnixStream;

    fn mount_share() {
        fs::create_dir_all("/mnt/shared").unwrap();
        let source = CString::new(TAG).unwrap();
        let target = CString::new("/mnt/shared").unwrap();
        let fstype = CString::new("virtiofs").unwrap();
        let ret = unsafe {
            libc::mount(
                source.as_ptr(),
                target.as_ptr(),
                fstype.as_ptr(),
                0,
                std::ptr::null(),
            )
        };
        assert_eq!(ret, 0, "mount: {}", std::io::Error::last_os_error());
    }

    fn refused<T: std::fmt::Debug>(result: std::io::Result<T>) {
        assert_eq!(result.unwrap_err().raw_os_error(), Some(EACCES));
    }

    fn append(path: &str) -> std::io::Result<fs::File> {
        fs::OpenOptions::new().append(true).open(path)
    }

    fn open_at(dir: &fs::File, name: &str, flags: i32) -> std::io::Result<fs::File> {
        let name = CString::new(name).unwrap();
        let fd = unsafe {
            libc::openat(
                dir.as_raw_fd(),
                name.as_ptr(),
                flags | libc::O_CLOEXEC,
                0o644,
            )
        };
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(unsafe { fs::File::from_raw_fd(fd) })
    }

    /// Tell the host the guest is done with a step, and wait for it to take the next one.
    fn step(control: &mut UnixStream) {
        control.write_all(&[1]).unwrap();
        control.read_exact(&mut [0]).unwrap();
    }

    impl Test for TestVirtiofsPolicy {
        fn in_guest(self: Box<Self>) {
            mount_share();

            // A hard link made on the host is the same file under another name. Tried
            // first, so that the host's descriptor of the file has the other name, and
            // only its identity tells.
            refused(append("/mnt/shared/free/alias"));
            refused(fs::write("/mnt/shared/protected/new", "x"));
            refused(append("/mnt/shared/protected/file"));
            refused(fs::remove_file("/mnt/shared/protected/file"));
            refused(fs::create_dir("/mnt/shared/protected/dir"));
            refused(fs::rename(
                "/mnt/shared/protected/file",
                "/mnt/shared/free/file",
            ));
            fs::write("/mnt/shared/free/new", "x").unwrap();
            fs::rename("/mnt/shared/free/new", "/mnt/shared/free/renamed").unwrap();
            refused(fs::rename(
                "/mnt/shared/free/renamed",
                "/mnt/shared/protected/renamed",
            ));
            // A symlink leads where it points, and a hard link made by the guest is
            // refused like a rename.
            symlink("../protected/file", "/mnt/shared/free/to-file").unwrap();
            refused(append("/mnt/shared/free/to-file"));
            symlink("../protected/via-link", "/mnt/shared/free/to-new").unwrap();
            refused(fs::write("/mnt/shared/free/to-new", "x"));
            refused(fs::hard_link(
                "/mnt/shared/protected/file",
                "/mnt/shared/free/linked",
            ));
            assert_eq!(
                fs::read_to_string("/mnt/shared/protected/file").unwrap(),
                "kept"
            );

            let sock = socket(
                AddressFamily::Vsock,
                SockType::Stream,
                SockFlag::empty(),
                None,
            )
            .unwrap();
            connect(
                sock.as_raw_fd(),
                &VsockAddr::new(VMADDR_CID_HOST, CONTROL_PORT),
            )
            .unwrap();
            let mut control = UnixStream::from(sock);
            step(&mut control);

            fs::write("/mnt/shared/protected/new", "x").unwrap();
            append("/mnt/shared/protected/file")
                .unwrap()
                .write_all(b", appended")
                .unwrap();
            assert_eq!(
                fs::read_to_string("/mnt/shared/protected/file").unwrap(),
                "kept, appended"
            );
            fs::create_dir("/mnt/shared/protected/dir").unwrap();
            let dir = fs::File::open("/mnt/shared/protected").unwrap();
            step(&mut control);

            // Each open is checked when it is made, through a directory opened before too.
            refused(open_at(&dir, "late", libc::O_CREAT | libc::O_WRONLY));
            refused(open_at(&dir, "new", libc::O_WRONLY));
            step(&mut control);
            println!("OK");
        }
    }
}
