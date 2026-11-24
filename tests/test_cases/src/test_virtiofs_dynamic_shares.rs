#![cfg(any(feature = "host", target_os = "linux"))]

//! A virtiofs device made with `FsDevice::new_dynamic` shows directories that
//! the host adds while the guest runs and hides those it removes, while a file
//! the guest opened before the removal stays readable. The guest can't move or
//! link files from one share into another, nor create anything next to them.

use macros::{guest, host};

pub struct TestVirtiofsDynamicShares;

/// The guest connects out on this port to synchronize with the host.
const CONTROL_PORT: u32 = 1234;
const TAG: &str = "dynamic";

const FIRST: &str = "in the first share";
const SECOND: &str = "in the second share";

#[host]
mod host {
    use super::*;
    #[cfg(feature = "dynamic-linking")]
    use krun::{DeviceManager, ffi::Symbol};

    use std::fs;
    use std::io::{Read, Write};
    use std::os::unix::net::UnixListener;
    use std::path::PathBuf;
    use std::thread;

    use crate::common::{build_init_config, setup_standard_devices};
    use crate::{ShouldRun, Test, TestOutcome, TestSetup};

    fn run(listener: UnixListener, shares: krun::DynamicShares, second: PathBuf) {
        let (mut control, _) = listener.accept().unwrap();
        // The guest has opened a file in the first share.
        control.read_exact(&mut [0]).unwrap();
        shares
            .add_read_only("second", second.to_str().unwrap())
            .unwrap();
        control.write_all(&[1]).unwrap();
        // The guest has tried to move files between the shares.
        control.read_exact(&mut [0]).unwrap();
        shares.remove("first").unwrap();
        control.write_all(&[1]).unwrap();
        // Keep the connection open until the guest is done.
        let _ = control.read(&mut [0]);
    }

    impl Test for TestVirtiofsDynamicShares {
        fn should_run(&self) -> ShouldRun {
            #[cfg(feature = "dynamic-linking")]
            if !Symbol::KrunFsDeviceNewDynamic.is_available() {
                return ShouldRun::No("feature not enabled in this libkrun build");
            }
            ShouldRun::Yes
        }

        fn timeout_secs(&self) -> u64 {
            30
        }

        fn start_vm(self: Box<Self>, test_setup: TestSetup) -> anyhow::Result<()> {
            let first = test_setup.tmp_dir.join("first");
            let second = test_setup.tmp_dir.join("second");
            fs::create_dir(&first)?;
            fs::create_dir(&second)?;
            fs::write(first.join("file"), FIRST)?;
            fs::write(second.join("file"), SECOND)?;

            let fs_dev = krun::FsDevice::new_dynamic(TAG)
                .map_err(|e| anyhow::anyhow!("FsDevice::new_dynamic: {e:?}"))?;
            let shares = fs_dev
                .dynamic_shares()
                .map_err(|e| anyhow::anyhow!("dynamic_shares: {e:?}"))?;
            shares
                .add("first", first.to_str().unwrap())
                .map_err(|e| anyhow::anyhow!("add: {e:?}"))?;

            let control_sock = test_setup.tmp_dir.join("control.sock");
            let listener = UnixListener::bind(&control_sock)?;
            thread::spawn(move || run(listener, shares, second));

            let init_config = build_init_config(&test_setup.test_case, &[]);
            let stdin = std::io::stdin();
            let stdout = std::io::stdout();
            let stderr = std::io::stderr();
            let (mut devices, payload) =
                setup_standard_devices(&test_setup, &init_config, &stdin, &stdout, &stderr)?;
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
            // The guest only read the first share before it was removed, and the
            // second is read-only: both hold only what the host put there.
            for (share, contents) in [("first", FIRST), ("second", SECOND)] {
                let dir = test_setup.tmp_dir.join(share);
                let names: Vec<_> = fs::read_dir(&dir)
                    .unwrap()
                    .map(|entry| entry.unwrap().file_name())
                    .collect();
                let file = fs::read_to_string(dir.join("file")).unwrap();
                if names != ["file"] || file != contents {
                    return TestOutcome::Fail(format!("the guest changed {share}: {names:?}"));
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
    use std::io::{Read, Seek, Write};
    use std::os::fd::AsRawFd;
    use std::os::unix::net::UnixStream;

    fn mount_share() {
        fs::create_dir_all("/mnt/dynamic").unwrap();
        let source = CString::new(TAG).unwrap();
        let target = CString::new("/mnt/dynamic").unwrap();
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

    fn errno<T: std::fmt::Debug>(result: std::io::Result<T>) -> Option<i32> {
        result.unwrap_err().raw_os_error()
    }

    fn entries() -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir("/mnt/dynamic")
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        names
    }

    impl Test for TestVirtiofsDynamicShares {
        fn in_guest(self: Box<Self>) {
            mount_share();

            assert_eq!(entries(), ["first"]);
            assert!(fs::metadata("/mnt/dynamic/second").is_err());
            let mut open = fs::File::open("/mnt/dynamic/first/file").unwrap();

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
            control.write_all(&[1]).unwrap();
            control.read_exact(&mut [0]).unwrap();

            assert_eq!(entries(), ["first", "second"]);
            for (from, to) in [("first", "second"), ("second", "first")] {
                let file = format!("/mnt/dynamic/{from}/file");
                let moved = format!("/mnt/dynamic/{to}/moved");
                assert_eq!(errno(fs::rename(&file, &moved)), Some(libc::EXDEV));
                assert_eq!(errno(fs::hard_link(&file, &moved)), Some(libc::EXDEV));
            }
            assert_eq!(
                errno(fs::write("/mnt/dynamic/new", "x")),
                Some(libc::EACCES)
            );
            assert_eq!(
                errno(fs::create_dir("/mnt/dynamic/dir")),
                Some(libc::EACCES)
            );
            control.write_all(&[1]).unwrap();
            control.read_exact(&mut [0]).unwrap();

            assert_eq!(entries(), ["second"]);
            assert_eq!(
                fs::read_to_string("/mnt/dynamic/second/file").unwrap(),
                SECOND
            );
            let err = fs::write("/mnt/dynamic/second/new", "x").unwrap_err();
            assert_eq!(err.raw_os_error(), Some(libc::EROFS));
            let err = fs::metadata("/mnt/dynamic/first").unwrap_err();
            assert_eq!(err.raw_os_error(), Some(libc::ENOENT));
            assert_eq!(
                errno(fs::write("/mnt/dynamic/first/new", "x")),
                Some(libc::ENOENT)
            );

            let mut contents = String::new();
            open.rewind().unwrap();
            open.read_to_string(&mut contents).unwrap();
            assert_eq!(contents, FIRST, "a file opened before the removal");

            control.write_all(&[1]).unwrap();
            println!("OK");
        }
    }
}
