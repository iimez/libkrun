#![cfg(all(
    any(feature = "host", target_os = "linux"),
    not(feature = "dynamic-linking")
))]

//! A virtiofs policy that defers a request holds it unanswered while the
//! guest goes on using the share, then answers it when the host says so: the
//! guest's call returns what the policy decided, an open that goes through or
//! one refused with its errno.

use macros::{guest, host};

pub struct TestVirtiofsDeferred;

/// The guest connects out on this port to synchronize with the host.
const CONTROL_PORT: u32 = 1234;
const TAG: &str = "shared";
const EACCES: i32 = 13;
/// The files whose opens for writing the policy defers.
const HELD: [&str; 2] = ["allowed", "denied"];

#[host]
mod host {
    use super::*;

    use std::collections::HashMap;
    use std::fs;
    use std::io::{Read, Write};
    use std::os::unix::net::UnixListener;
    use std::path::Path;
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::sync::{Arc, Mutex};
    use std::thread;

    use crate::common::{build_init_config, setup_standard_devices};
    use crate::{Test, TestOutcome, TestSetup};

    /// Defers each open for writing of a file in `HELD` until the host answers it.
    struct DeferWrites {
        deferred: Mutex<Sender<(String, krun::FsRequestId, krun::FsDeferral)>>,
        answers: Mutex<HashMap<krun::FsRequestId, krun::FsVerdict>>,
    }

    impl krun::FsPolicy for DeferWrites {
        fn check(&self, request: &krun::FsRequest) -> krun::FsVerdict {
            if let Some(verdict) = self.answers.lock().unwrap().remove(&request.id()) {
                return verdict;
            }
            let path = request.object().path().and_then(Path::to_str);
            let writes = request.flags() & 3 != 0;
            match path {
                Some(name)
                    if request.op() == krun::FsOp::Open && writes && HELD.contains(&name) =>
                {
                    let deferred = (name.to_owned(), request.id(), request.deferral());
                    self.deferred.lock().unwrap().send(deferred).unwrap();
                    krun::FsVerdict::Defer
                }
                _ => krun::FsVerdict::Allow,
            }
        }
    }

    fn run(
        listener: UnixListener,
        policy: Arc<DeferWrites>,
        deferred: Receiver<(String, krun::FsRequestId, krun::FsDeferral)>,
    ) {
        let (mut control, _) = listener.accept().unwrap();
        for (file, verdict) in HELD
            .iter()
            .zip([krun::FsVerdict::Allow, krun::FsVerdict::Deny(EACCES)])
        {
            // The guest is opening the file.
            control.read_exact(&mut [0]).unwrap();
            let (name, id, deferral) = deferred.recv().unwrap();
            assert_eq!(name, *file);
            control.write_all(&[1]).unwrap();
            // The guest has used the share while the open waits.
            control.read_exact(&mut [0]).unwrap();
            policy.answers.lock().unwrap().insert(id, verdict);
            deferral.decided();
            control.write_all(&[1]).unwrap();
        }
        // Keep the connection open until the guest is done.
        let _ = control.read(&mut [0]);
    }

    impl Test for TestVirtiofsDeferred {
        fn timeout_secs(&self) -> u64 {
            30
        }

        fn start_vm(self: Box<Self>, test_setup: TestSetup) -> anyhow::Result<()> {
            let shared = test_setup.tmp_dir.join("shared");
            fs::create_dir_all(&shared)?;
            for file in HELD {
                fs::write(shared.join(file), "before")?;
            }

            let (send, deferred) = mpsc::channel();
            let policy = Arc::new(DeferWrites {
                deferred: Mutex::new(send),
                answers: Mutex::new(HashMap::new()),
            });
            let control_sock = test_setup.tmp_dir.join("control.sock");
            let listener = UnixListener::bind(&control_sock)?;
            let answering = policy.clone();
            thread::spawn(move || run(listener, answering, deferred));

            let init_config = build_init_config(&test_setup.test_case, &[]);
            let stdin = std::io::stdin();
            let stdout = std::io::stdout();
            let stderr = std::io::stderr();
            let (mut devices, payload) =
                setup_standard_devices(&test_setup, &init_config, &stdin, &stdout, &stderr)?;

            let mut fs_dev = krun::FsDevice::new(TAG, shared.to_str().unwrap())
                .map_err(|e| anyhow::anyhow!("FsDevice::new: {e:?}"))?;
            fs_dev
                .set_policy(policy)
                .map_err(|e| anyhow::anyhow!("set_policy: {e:?}"))?;
            devices.add(fs_dev);

            let mut vsock = krun::VsockDevice::new(3, krun::TsiFlags::empty())
                .map_err(|e| anyhow::anyhow!("VsockDevice: {e:?}"))?;
            vsock.add_unix_port(CONTROL_PORT, control_sock.to_str().unwrap(), false);
            devices.add(vsock);

            let vmm = krun::VmmBuilder::new()
                .vcpus(2)
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
            let shared = test_setup.tmp_dir.join("shared");
            let contents = HELD.map(|file| fs::read_to_string(shared.join(file)).unwrap());
            if contents != ["written", "before"] {
                return TestOutcome::Fail(format!(
                    "expected the allowed file written and the denied one unchanged, got \
                     {contents:?}"
                ));
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
    use std::os::fd::AsRawFd;
    use std::os::unix::net::UnixStream;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

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

    /// Tell the host the guest is done with a step, and wait for it to take the next one.
    fn step(control: &mut UnixStream) {
        control.write_all(&[1]).unwrap();
        control.read_exact(&mut [0]).unwrap();
    }

    impl Test for TestVirtiofsDeferred {
        fn in_guest(self: Box<Self>) {
            mount_share();
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

            for file in HELD {
                let path = format!("/mnt/shared/{file}");
                // Looked up first, so that the open below is an open of an existing file,
                // which holds no lock of the directory in the guest.
                fs::metadata(&path).unwrap();
                let done = Arc::new(AtomicBool::new(false));
                let opener = {
                    let done = done.clone();
                    thread::spawn(move || {
                        let result = fs::OpenOptions::new()
                            .write(true)
                            .truncate(true)
                            .open(&path)
                            .and_then(|mut file| file.write_all(b"written"));
                        done.store(true, Ordering::SeqCst);
                        result
                    })
                };
                step(&mut control);

                fs::write(format!("/mnt/shared/beside-{file}"), "x").unwrap();
                assert_eq!(
                    fs::read_to_string(format!("/mnt/shared/beside-{file}")).unwrap(),
                    "x"
                );
                assert_eq!(
                    fs::read_to_string(format!("/mnt/shared/{file}")).unwrap(),
                    "before"
                );
                assert!(
                    !done.load(Ordering::SeqCst),
                    "the open of {file} returned before the host answered it"
                );
                step(&mut control);

                let result = opener.join().unwrap();
                match file {
                    "allowed" => result.unwrap(),
                    _ => assert_eq!(result.unwrap_err().raw_os_error(), Some(EACCES)),
                }
            }
            println!("OK");
        }
    }
}
