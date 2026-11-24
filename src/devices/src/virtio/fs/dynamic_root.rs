use std::collections::BTreeMap;
use std::ffi::{CStr, CString, OsStr};
use std::fs::File;
use std::io;
use std::mem;
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::time::Duration;

#[cfg(target_os = "macos")]
use crossbeam_channel::Sender;
#[cfg(target_os = "macos")]
use utils::worker_message::WorkerMessage;

use super::filesystem::{
    Context, DirEntry, Entry, Extensions, FileSystem, FsOptions, GetxattrReply, ListxattrReply,
    OpenOptions, SetattrValid, ZeroCopyReader, ZeroCopyWriter,
};
use super::fuse;
use super::inode_alloc::InodeAllocator;
use super::passthrough::{self, PassthroughFs};
use super::policy::{HostNode, HostNodes};
use super::read_only::PassthroughFsRo;
use crate::virtio::bindings;
use crate::virtio::linux_errno;

type Inode = u64;
type Handle = u64;

// Share `id` owns the node IDs from `id << SHARE_SHIFT`, so a node ID names its share without a
// table. Handles aren't translated: a request with a handle also carries the node ID it was
// opened on, which picks the passthrough that allocated the handle.
const SHARE_SHIFT: u32 = 40;
const MAX_SHARES: usize = (1 << (u64::BITS - SHARE_SHIFT)) - 1;
const ROOT_BLKSIZE: u32 = 4096;

fn share_root(id: usize) -> Inode {
    ((id as u64) << SHARE_SHIFT) + 1
}

fn share_name(name: &str) -> io::Result<CString> {
    if name.is_empty() || name == "." || name == ".." || name.contains('/') {
        return Err(io::Error::from_raw_os_error(libc::EINVAL));
    }
    CString::new(name).map_err(|_| io::Error::from_raw_os_error(libc::EINVAL))
}

#[derive(Clone)]
struct ShareConfig {
    host_path: String,
    read_only: bool,
}

/// The entries the embedder asked for.
#[derive(Default)]
struct ShareList {
    shares: BTreeMap<CString, ShareConfig>,
    served: Weak<ServedShares>,
}

/// The entries of a dynamic virtio-fs root, which the embedder changes while the guest runs.
///
/// Clones refer to the same entries. Every method may be called from any thread, before the
/// device is activated or while it serves the guest. An entry whose directory can't be served
/// when the device is activated or the guest mounts it is logged and dropped.
#[derive(Clone, Default)]
pub struct DynamicShares {
    list: Arc<Mutex<ShareList>>,
}

impl DynamicShares {
    pub fn new() -> Self {
        Self::default()
    }

    /// Show `host_path` in the guest as the directory `name` at the root of the filesystem.
    ///
    /// Fails with `EINVAL` if `name` is not a single path component, with `EEXIST` if an entry
    /// has that name, and with the error of opening `host_path` if this process can't open it
    /// for reading.
    pub fn add(&self, name: &str, host_path: &str, read_only: bool) -> io::Result<()> {
        let name = share_name(name)?;
        let mut list = self.list.lock().unwrap();
        if list.shares.contains_key(&name) {
            return Err(io::Error::from_raw_os_error(libc::EEXIST));
        }
        if !File::open(host_path)?.metadata()?.is_dir() {
            return Err(io::Error::from_raw_os_error(libc::ENOTDIR));
        }
        let config = ShareConfig {
            host_path: host_path.to_owned(),
            read_only,
        };
        if let Some(shares) = list.served.upgrade() {
            shares.attach(name.clone(), &config)?;
        }
        list.shares.insert(name, config);
        Ok(())
    }

    /// Hide the entry `name` from the guest from now on.
    ///
    /// Files and directories under it that the guest holds open keep working until it closes
    /// them. Fails with `ENOENT` if there is no entry with that name.
    pub fn remove(&self, name: &str) -> io::Result<()> {
        let name = CString::new(name).map_err(|_| io::Error::from_raw_os_error(libc::ENOENT))?;
        let mut list = self.list.lock().unwrap();
        list.shares
            .remove(&name)
            .ok_or_else(|| io::Error::from_raw_os_error(libc::ENOENT))?;
        if let Some(shares) = list.served.upgrade() {
            shares.detach(&name);
        }
        Ok(())
    }
}

enum ShareFs {
    ReadWrite(PassthroughFs),
    ReadOnly(PassthroughFsRo),
}

macro_rules! with_fs {
    ($share:expr, $fs:ident => $call:expr) => {
        match &$share.fs {
            ShareFs::ReadWrite($fs) => $call,
            ShareFs::ReadOnly($fs) => $call,
        }
    };
}

struct Share {
    id: usize,
    name: CString,
    fs: ShareFs,
    removed: AtomicBool,
    /// The guest's lookups of the share's root that it hasn't forgotten, which the passthrough
    /// doesn't count.
    root_lookups: AtomicU64,
}

impl Share {
    fn looked_up_root(&self) {
        self.root_lookups.fetch_add(1, Ordering::SeqCst);
    }

    fn forgot_root(&self, count: u64) {
        let _ = self
            .root_lookups
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                Some(n.saturating_sub(count))
            });
    }
}

/// The shares served by one activation of the device.
struct ServedShares {
    template: passthrough::Config,
    /// Share `id` is at index `id - 1`. Removed shares stay, for the node IDs the guest holds.
    shares: RwLock<Vec<Arc<Share>>>,
    names: RwLock<BTreeMap<CString, Arc<Share>>>,
    /// What the guest offered at `FUSE_INIT`, once it has; shares added later are initialized
    /// with it.
    offered: Mutex<Option<FsOptions>>,
}

impl ServedShares {
    fn attach(&self, name: CString, config: &ShareConfig) -> io::Result<()> {
        // Held until the share is visible, so that `init` can't run in between.
        let offered = self.offered.lock().unwrap();
        let mut shares = self.shares.write().unwrap();
        let id = shares.len() + 1;
        if id > MAX_SHARES {
            return Err(io::Error::from_raw_os_error(libc::ENOSPC));
        }
        let cfg = passthrough::Config {
            root_dir: config.host_path.clone(),
            ..self.template.clone()
        };
        let inode_alloc = Arc::new(InodeAllocator::starting_at(share_root(id) + 1));
        let fs = if config.read_only {
            ShareFs::ReadOnly(PassthroughFsRo::new(cfg, inode_alloc)?)
        } else {
            ShareFs::ReadWrite(PassthroughFs::new(cfg, inode_alloc)?)
        };
        let share = Arc::new(Share {
            id,
            name: name.clone(),
            fs,
            removed: AtomicBool::new(false),
            root_lookups: AtomicU64::new(0),
        });
        if let Some(offered) = *offered {
            with_fs!(share, fs => fs.init(offered))?;
        }
        shares.push(share.clone());
        self.names.write().unwrap().insert(name, share);
        Ok(())
    }

    fn detach(&self, name: &CStr) {
        if let Some(share) = self.names.write().unwrap().remove(name) {
            share.removed.store(true, Ordering::SeqCst);
        }
    }
}

struct RootDirEntry {
    name: CString,
    ino: u64,
}

/// A `FileSystem` whose root lists the entries of a [`DynamicShares`].
pub struct DynamicRootFs {
    list: Arc<Mutex<ShareList>>,
    shares: Arc<ServedShares>,
    /// What each open handle of the root lists, as it was when it was opened.
    root_dirs: Mutex<BTreeMap<Handle, Vec<RootDirEntry>>>,
    next_root_handle: AtomicU64,
    uid: u32,
    gid: u32,
}

impl DynamicRootFs {
    /// Serve the entries of `shares`, each through a passthrough configured as `template`
    /// apart from its root directory.
    pub fn new(template: passthrough::Config, shares: &DynamicShares) -> Self {
        let template = passthrough::Config {
            // Negotiated once for the whole filesystem; the shares can't each choose.
            writeback: false,
            // Each share opens a descriptor of its own.
            proc_sfd_rawfd: None,
            export_table: None,
            ..template
        };
        let served = Arc::new(ServedShares {
            template,
            shares: RwLock::new(Vec::new()),
            names: RwLock::new(BTreeMap::new()),
            offered: Mutex::new(None),
        });
        let mut list = shares.list.lock().unwrap();
        list.shares
            .retain(|name, config| match served.attach(name.clone(), config) {
                Ok(()) => true,
                Err(e) => {
                    error!("virtio-fs: can't share {}: {e}", config.host_path);
                    false
                }
            });
        list.served = Arc::downgrade(&served);
        Self {
            list: shares.list.clone(),
            shares: served,
            root_dirs: Mutex::new(BTreeMap::new()),
            next_root_handle: AtomicU64::new(1),
            // SAFETY: these calls can't fail and touch no memory.
            uid: unsafe { libc::getuid() },
            gid: unsafe { libc::getgid() },
        }
    }

    /// The share a node ID belongs to, and the node ID its passthrough knows it by.
    fn route(&self, inode: Inode) -> io::Result<(Arc<Share>, Inode)> {
        let id = (inode >> SHARE_SHIFT) as usize;
        if id == 0 || inode < share_root(id) {
            return Err(linux_errno::ebadf());
        }
        let share = self
            .shares
            .shares
            .read()
            .unwrap()
            .get(id - 1)
            .cloned()
            .ok_or_else(linux_errno::ebadf)?;
        let inner = if inode == share_root(id) {
            // The root's node ID is known without a lookup, so a removed share would stay
            // reachable through it; only a guest that still holds it may keep using it.
            if share.removed.load(Ordering::SeqCst)
                && share.root_lookups.load(Ordering::SeqCst) == 0
            {
                return Err(linux_errno::ebadf());
            }
            fuse::ROOT_ID
        } else {
            inode
        };
        Ok((share, inner))
    }

    /// As `route`, for the two node IDs of an operation that can't cross shares.
    fn route_pair(&self, a: Inode, b: Inode) -> io::Result<(Arc<Share>, Inode, Inode)> {
        if a == fuse::ROOT_ID || b == fuse::ROOT_ID {
            return Err(linux_errno::eacces());
        }
        let (share, a) = self.route(a)?;
        let (other, b) = self.route(b)?;
        if share.id != other.id {
            return Err(linux_errno::exdev());
        }
        Ok((share, a, b))
    }

    fn guest_entry(share: &Share, mut entry: Entry) -> Entry {
        if entry.inode == fuse::ROOT_ID {
            entry.inode = share_root(share.id);
            share.looked_up_root();
        }
        entry
    }

    fn root_attr(&self) -> bindings::stat64 {
        // SAFETY: stat64 is plain data, for which all-zero bytes are a valid value.
        let mut st: bindings::stat64 = unsafe { mem::zeroed() };
        st.st_ino = fuse::ROOT_ID;
        st.st_mode = (libc::S_IFDIR | 0o555) as _;
        st.st_nlink = 2;
        st.st_uid = self.uid;
        st.st_gid = self.gid;
        st.st_blksize = ROOT_BLKSIZE as _;
        st
    }

    fn root_listing(&self, ctx: Context) -> Vec<RootDirEntry> {
        let mut listing = vec![
            RootDirEntry {
                name: c".".into(),
                ino: fuse::ROOT_ID,
            },
            RootDirEntry {
                name: c"..".into(),
                ino: fuse::ROOT_ID,
            },
        ];
        for (name, share) in self.shares.names.read().unwrap().iter() {
            let ino = with_fs!(share, fs => fs.getattr(ctx, fuse::ROOT_ID, None))
                .map_or(share_root(share.id), |(st, _)| st.st_ino);
            listing.push(RootDirEntry {
                name: name.clone(),
                ino,
            });
        }
        listing
    }
}

impl HostNodes for DynamicRootFs {
    fn host_node(&self, inode: Inode) -> io::Result<HostNode> {
        if inode == fuse::ROOT_ID {
            return Err(linux_errno::enoent());
        }
        let (share, inode) = self.route(inode)?;
        with_fs!(share, fs => fs.host_node(inode))
    }

    fn host_child(&self, parent: Inode, name: &CStr) -> io::Result<Option<HostNode>> {
        if parent == fuse::ROOT_ID {
            let share = self.shares.names.read().unwrap().get(name).cloned();
            return share.map_or(Ok(None), |share| {
                with_fs!(share, fs => fs.host_node(fuse::ROOT_ID)).map(Some)
            });
        }
        let (share, parent) = self.route(parent)?;
        with_fs!(share, fs => fs.host_child(parent, name))
    }

    fn host_path(&self, inode: Inode) -> io::Result<Option<PathBuf>> {
        if inode == fuse::ROOT_ID {
            return Ok(Some(PathBuf::new()));
        }
        let (share, inode) = self.route(inode)?;
        let visible = self
            .shares
            .names
            .read()
            .unwrap()
            .get(&share.name)
            .is_some_and(|named| named.id == share.id);
        if !visible {
            return Ok(None);
        }
        let path = with_fs!(share, fs => fs.host_path(inode))?;
        Ok(path.map(|path| PathBuf::from(OsStr::from_bytes(share.name.to_bytes())).join(path)))
    }
}

macro_rules! not_on_root {
    ($inode:expr) => {
        if $inode == fuse::ROOT_ID {
            return Err(linux_errno::eacces());
        }
    };
}

impl FileSystem for DynamicRootFs {
    type Inode = Inode;
    type Handle = Handle;

    fn init(&self, capable: FsOptions) -> io::Result<FsOptions> {
        // In the order `DynamicShares::add` takes them.
        let mut list = self.list.lock().unwrap();
        let mut offered = self.shares.offered.lock().unwrap();
        let mut names = self.shares.names.write().unwrap();
        names.retain(
            |name, share| match with_fs!(share, fs => fs.init(capable)) {
                Ok(_) => true,
                Err(e) => {
                    error!("virtio-fs: can't share {name:?}: {e}");
                    list.shares.remove(name);
                    false
                }
            },
        );
        *offered = Some(capable);
        Ok(FsOptions::DO_READDIRPLUS | FsOptions::READDIRPLUS_AUTO)
    }

    fn destroy(&self) {
        for share in self.shares.shares.read().unwrap().iter() {
            with_fs!(share, fs => fs.destroy());
        }
    }

    fn lookup(&self, ctx: Context, parent: Inode, name: &CStr) -> io::Result<Entry> {
        if parent == fuse::ROOT_ID {
            let share = self
                .shares
                .names
                .read()
                .unwrap()
                .get(name)
                .cloned()
                .ok_or_else(linux_errno::enoent)?;
            let (attr, _) = with_fs!(share, fs => fs.getattr(ctx, fuse::ROOT_ID, None))?;
            share.looked_up_root();
            return Ok(Entry {
                inode: share_root(share.id),
                generation: 0,
                attr,
                attr_flags: 0,
                attr_timeout: Duration::ZERO,
                entry_timeout: Duration::ZERO,
            });
        }
        let (share, parent) = self.route(parent)?;
        let entry = with_fs!(share, fs => fs.lookup(ctx, parent, name))?;
        Ok(Self::guest_entry(&share, entry))
    }

    fn forget(&self, ctx: Context, inode: Inode, count: u64) {
        let Ok((share, inode)) = self.route(inode) else {
            return;
        };
        if inode == fuse::ROOT_ID {
            share.forgot_root(count);
        } else {
            with_fs!(share, fs => fs.forget(ctx, inode, count));
        }
    }

    fn getattr(
        &self,
        ctx: Context,
        inode: Inode,
        handle: Option<Handle>,
    ) -> io::Result<(bindings::stat64, Duration)> {
        if inode == fuse::ROOT_ID {
            return Ok((self.root_attr(), Duration::ZERO));
        }
        let (share, inode) = self.route(inode)?;
        with_fs!(share, fs => fs.getattr(ctx, inode, handle))
    }

    fn setattr(
        &self,
        ctx: Context,
        inode: Inode,
        attr: bindings::stat64,
        handle: Option<Handle>,
        valid: SetattrValid,
    ) -> io::Result<(bindings::stat64, Duration)> {
        not_on_root!(inode);
        let (share, inode) = self.route(inode)?;
        with_fs!(share, fs => fs.setattr(ctx, inode, attr, handle, valid))
    }

    fn readlink(&self, ctx: Context, inode: Inode) -> io::Result<Vec<u8>> {
        if inode == fuse::ROOT_ID {
            return Err(linux_errno::einval());
        }
        let (share, inode) = self.route(inode)?;
        with_fs!(share, fs => fs.readlink(ctx, inode))
    }

    fn symlink(
        &self,
        ctx: Context,
        linkname: &CStr,
        parent: Inode,
        name: &CStr,
        extensions: Extensions,
    ) -> io::Result<Entry> {
        not_on_root!(parent);
        let (share, parent) = self.route(parent)?;
        let entry = with_fs!(share, fs => fs.symlink(ctx, linkname, parent, name, extensions))?;
        Ok(Self::guest_entry(&share, entry))
    }

    fn mknod(
        &self,
        ctx: Context,
        parent: Inode,
        name: &CStr,
        mode: u32,
        rdev: u32,
        umask: u32,
        extensions: Extensions,
    ) -> io::Result<Entry> {
        not_on_root!(parent);
        let (share, parent) = self.route(parent)?;
        let entry =
            with_fs!(share, fs => fs.mknod(ctx, parent, name, mode, rdev, umask, extensions))?;
        Ok(Self::guest_entry(&share, entry))
    }

    fn mkdir(
        &self,
        ctx: Context,
        parent: Inode,
        name: &CStr,
        mode: u32,
        umask: u32,
        extensions: Extensions,
    ) -> io::Result<Entry> {
        not_on_root!(parent);
        let (share, parent) = self.route(parent)?;
        let entry = with_fs!(share, fs => fs.mkdir(ctx, parent, name, mode, umask, extensions))?;
        Ok(Self::guest_entry(&share, entry))
    }

    fn unlink(&self, ctx: Context, parent: Inode, name: &CStr) -> io::Result<()> {
        not_on_root!(parent);
        let (share, parent) = self.route(parent)?;
        with_fs!(share, fs => fs.unlink(ctx, parent, name))
    }

    fn rmdir(&self, ctx: Context, parent: Inode, name: &CStr) -> io::Result<()> {
        not_on_root!(parent);
        let (share, parent) = self.route(parent)?;
        with_fs!(share, fs => fs.rmdir(ctx, parent, name))
    }

    fn rename(
        &self,
        ctx: Context,
        olddir: Inode,
        oldname: &CStr,
        newdir: Inode,
        newname: &CStr,
        flags: u32,
    ) -> io::Result<()> {
        let (share, olddir, newdir) = self.route_pair(olddir, newdir)?;
        with_fs!(share, fs => fs.rename(ctx, olddir, oldname, newdir, newname, flags))
    }

    fn link(
        &self,
        ctx: Context,
        inode: Inode,
        newparent: Inode,
        newname: &CStr,
    ) -> io::Result<Entry> {
        let (share, inode, newparent) = self.route_pair(inode, newparent)?;
        let entry = with_fs!(share, fs => fs.link(ctx, inode, newparent, newname))?;
        Ok(Self::guest_entry(&share, entry))
    }

    fn open(
        &self,
        ctx: Context,
        inode: Inode,
        kill_priv: bool,
        flags: u32,
    ) -> io::Result<(Option<Handle>, OpenOptions)> {
        if inode == fuse::ROOT_ID {
            return Err(linux_errno::eisdir());
        }
        let (share, inode) = self.route(inode)?;
        with_fs!(share, fs => fs.open(ctx, inode, kill_priv, flags))
    }

    fn create(
        &self,
        ctx: Context,
        parent: Inode,
        name: &CStr,
        mode: u32,
        kill_priv: bool,
        flags: u32,
        umask: u32,
        extensions: Extensions,
    ) -> io::Result<(Entry, Option<Handle>, OpenOptions)> {
        not_on_root!(parent);
        let (share, parent) = self.route(parent)?;
        let (entry, handle, opts) = with_fs!(share, fs => fs.create(
            ctx, parent, name, mode, kill_priv, flags, umask, extensions
        ))?;
        Ok((Self::guest_entry(&share, entry), handle, opts))
    }

    fn read<W: io::Write + ZeroCopyWriter>(
        &self,
        ctx: Context,
        inode: Inode,
        handle: Handle,
        w: W,
        size: u32,
        offset: u64,
        lock_owner: Option<u64>,
        flags: u32,
    ) -> io::Result<usize> {
        if inode == fuse::ROOT_ID {
            return Err(linux_errno::eisdir());
        }
        let (share, inode) = self.route(inode)?;
        with_fs!(share, fs => fs.read(ctx, inode, handle, w, size, offset, lock_owner, flags))
    }

    fn write<R: io::Read + ZeroCopyReader>(
        &self,
        ctx: Context,
        inode: Inode,
        handle: Handle,
        r: R,
        size: u32,
        offset: u64,
        lock_owner: Option<u64>,
        delayed_write: bool,
        kill_priv: bool,
        flags: u32,
    ) -> io::Result<usize> {
        not_on_root!(inode);
        let (share, inode) = self.route(inode)?;
        with_fs!(share, fs => fs.write(
            ctx,
            inode,
            handle,
            r,
            size,
            offset,
            lock_owner,
            delayed_write,
            kill_priv,
            flags,
        ))
    }

    fn flush(&self, ctx: Context, inode: Inode, handle: Handle, lock_owner: u64) -> io::Result<()> {
        if inode == fuse::ROOT_ID {
            return Ok(());
        }
        let (share, inode) = self.route(inode)?;
        with_fs!(share, fs => fs.flush(ctx, inode, handle, lock_owner))
    }

    fn fsync(&self, ctx: Context, inode: Inode, datasync: bool, handle: Handle) -> io::Result<()> {
        if inode == fuse::ROOT_ID {
            return Ok(());
        }
        let (share, inode) = self.route(inode)?;
        with_fs!(share, fs => fs.fsync(ctx, inode, datasync, handle))
    }

    fn fallocate(
        &self,
        ctx: Context,
        inode: Inode,
        handle: Handle,
        mode: u32,
        offset: u64,
        length: u64,
    ) -> io::Result<()> {
        not_on_root!(inode);
        let (share, inode) = self.route(inode)?;
        with_fs!(share, fs => fs.fallocate(ctx, inode, handle, mode, offset, length))
    }

    fn release(
        &self,
        ctx: Context,
        inode: Inode,
        flags: u32,
        handle: Handle,
        flush: bool,
        flock_release: bool,
        lock_owner: Option<u64>,
    ) -> io::Result<()> {
        if inode == fuse::ROOT_ID {
            return Ok(());
        }
        let (share, inode) = self.route(inode)?;
        with_fs!(share, fs => fs.release(
            ctx,
            inode,
            flags,
            handle,
            flush,
            flock_release,
            lock_owner,
        ))
    }

    fn statfs(&self, ctx: Context, inode: Inode) -> io::Result<bindings::statvfs64> {
        if inode == fuse::ROOT_ID {
            // SAFETY: statvfs64 is plain data, for which all-zero bytes are a valid value.
            let mut st: bindings::statvfs64 = unsafe { mem::zeroed() };
            st.f_bsize = ROOT_BLKSIZE.into();
            st.f_frsize = ROOT_BLKSIZE.into();
            st.f_namemax = 255;
            return Ok(st);
        }
        let (share, inode) = self.route(inode)?;
        with_fs!(share, fs => fs.statfs(ctx, inode))
    }

    fn setxattr(
        &self,
        ctx: Context,
        inode: Inode,
        name: &CStr,
        value: &[u8],
        flags: u32,
    ) -> io::Result<()> {
        not_on_root!(inode);
        let (share, inode) = self.route(inode)?;
        with_fs!(share, fs => fs.setxattr(ctx, inode, name, value, flags))
    }

    fn getxattr(
        &self,
        ctx: Context,
        inode: Inode,
        name: &CStr,
        size: u32,
    ) -> io::Result<GetxattrReply> {
        if inode == fuse::ROOT_ID {
            return Err(linux_errno::enodata());
        }
        let (share, inode) = self.route(inode)?;
        with_fs!(share, fs => fs.getxattr(ctx, inode, name, size))
    }

    fn listxattr(&self, ctx: Context, inode: Inode, size: u32) -> io::Result<ListxattrReply> {
        if inode == fuse::ROOT_ID {
            if size == 0 {
                return Ok(ListxattrReply::Count(0));
            }
            return Ok(ListxattrReply::Names(Vec::new()));
        }
        let (share, inode) = self.route(inode)?;
        with_fs!(share, fs => fs.listxattr(ctx, inode, size))
    }

    fn removexattr(&self, ctx: Context, inode: Inode, name: &CStr) -> io::Result<()> {
        not_on_root!(inode);
        let (share, inode) = self.route(inode)?;
        with_fs!(share, fs => fs.removexattr(ctx, inode, name))
    }

    fn opendir(
        &self,
        ctx: Context,
        inode: Inode,
        flags: u32,
    ) -> io::Result<(Option<Handle>, OpenOptions)> {
        if inode == fuse::ROOT_ID {
            let handle = self.next_root_handle.fetch_add(1, Ordering::Relaxed);
            let listing = self.root_listing(ctx);
            self.root_dirs.lock().unwrap().insert(handle, listing);
            return Ok((Some(handle), OpenOptions::empty()));
        }
        let (share, inode) = self.route(inode)?;
        with_fs!(share, fs => fs.opendir(ctx, inode, flags))
    }

    fn readdir<F>(
        &self,
        ctx: Context,
        inode: Inode,
        handle: Handle,
        size: u32,
        offset: u64,
        mut add_entry: F,
    ) -> io::Result<()>
    where
        F: FnMut(DirEntry) -> io::Result<usize>,
    {
        if inode == fuse::ROOT_ID {
            if size == 0 {
                return Ok(());
            }
            let root_dirs = self.root_dirs.lock().unwrap();
            let listing = root_dirs.get(&handle).ok_or_else(linux_errno::ebadf)?;
            for (index, entry) in listing.iter().enumerate().skip(offset as usize) {
                let added = add_entry(DirEntry {
                    ino: entry.ino,
                    offset: index as u64 + 1,
                    type_: libc::DT_DIR.into(),
                    name: entry.name.to_bytes(),
                })?;
                if added == 0 {
                    break;
                }
            }
            return Ok(());
        }
        let (share, inode) = self.route(inode)?;
        with_fs!(share, fs => fs.readdir(ctx, inode, handle, size, offset, add_entry))
    }

    fn readdirplus<F>(
        &self,
        ctx: Context,
        inode: Inode,
        handle: Handle,
        size: u32,
        offset: u64,
        mut add_entry: F,
    ) -> io::Result<()>
    where
        F: FnMut(DirEntry, Entry) -> io::Result<usize>,
    {
        if inode == fuse::ROOT_ID {
            // Names only: the guest looks each one up, and a name removed by then fails there.
            return self.readdir(ctx, inode, handle, size, offset, |dir_entry| {
                add_entry(dir_entry, Entry::without_attributes())
            });
        }
        let (share, inode) = self.route(inode)?;
        with_fs!(share, fs => fs.readdirplus(ctx, inode, handle, size, offset, |dir_entry, entry| {
            add_entry(dir_entry, Self::guest_entry(&share, entry))
        }))
    }

    fn fsyncdir(
        &self,
        ctx: Context,
        inode: Inode,
        datasync: bool,
        handle: Handle,
    ) -> io::Result<()> {
        if inode == fuse::ROOT_ID {
            return Ok(());
        }
        let (share, inode) = self.route(inode)?;
        with_fs!(share, fs => fs.fsyncdir(ctx, inode, datasync, handle))
    }

    fn releasedir(&self, ctx: Context, inode: Inode, flags: u32, handle: Handle) -> io::Result<()> {
        if inode == fuse::ROOT_ID {
            self.root_dirs.lock().unwrap().remove(&handle);
            return Ok(());
        }
        let (share, inode) = self.route(inode)?;
        with_fs!(share, fs => fs.releasedir(ctx, inode, flags, handle))
    }

    fn access(&self, ctx: Context, inode: Inode, mask: u32) -> io::Result<()> {
        if inode == fuse::ROOT_ID {
            if mask & libc::W_OK as u32 != 0 {
                return Err(linux_errno::eacces());
            }
            return Ok(());
        }
        let (share, inode) = self.route(inode)?;
        with_fs!(share, fs => fs.access(ctx, inode, mask))
    }

    fn lseek(
        &self,
        ctx: Context,
        inode: Inode,
        handle: Handle,
        offset: u64,
        whence: u32,
    ) -> io::Result<u64> {
        if inode == fuse::ROOT_ID {
            return Err(linux_errno::einval());
        }
        let (share, inode) = self.route(inode)?;
        with_fs!(share, fs => fs.lseek(ctx, inode, handle, offset, whence))
    }

    fn copyfilerange(
        &self,
        ctx: Context,
        inode_in: Inode,
        handle_in: Handle,
        offset_in: u64,
        inode_out: Inode,
        handle_out: Handle,
        offset_out: u64,
        len: u64,
        flags: u64,
    ) -> io::Result<usize> {
        let (share, inode_in, inode_out) = self.route_pair(inode_in, inode_out)?;
        with_fs!(share, fs => fs.copyfilerange(
            ctx, inode_in, handle_in, offset_in, inode_out, handle_out, offset_out, len, flags,
        ))
    }

    fn setupmapping(
        &self,
        _ctx: Context,
        _inode: Inode,
        _handle: Handle,
        _foffset: u64,
        _len: u64,
        _flags: u64,
        _moffset: u64,
        _host_shm_base: u64,
        _shm_size: u64,
        #[cfg(target_os = "macos")] _map_sender: &Option<Sender<WorkerMessage>>,
    ) -> io::Result<()> {
        Err(linux_errno::enosys())
    }

    fn removemapping(
        &self,
        _ctx: Context,
        _requests: Vec<fuse::RemovemappingOne>,
        _host_shm_base: u64,
        _shm_size: u64,
        #[cfg(target_os = "macos")] _map_sender: &Option<Sender<WorkerMessage>>,
    ) -> io::Result<()> {
        Err(linux_errno::enosys())
    }

    fn ioctl(
        &self,
        ctx: Context,
        inode: Inode,
        handle: Handle,
        flags: u32,
        cmd: u32,
        arg: u64,
        in_size: u32,
        out_size: u32,
        exit_code: &Arc<AtomicI32>,
    ) -> io::Result<Vec<u8>> {
        not_on_root!(inode);
        let (share, inode) = self.route(inode)?;
        with_fs!(share, fs => fs.ioctl(
            ctx, inode, handle, flags, cmd, arg, in_size, out_size, exit_code,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use super::*;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "libkrun-dynamic-root-{}-{name}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("file"), name).unwrap();
            Self(dir)
        }

        fn path(&self) -> &str {
            self.0.to_str().unwrap()
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn ctx() -> Context {
        // SAFETY: these calls can't fail and touch no memory.
        unsafe {
            Context {
                uid: libc::getuid(),
                gid: libc::getgid(),
                pid: 0,
            }
        }
    }

    fn mounted(shares: &DynamicShares) -> DynamicRootFs {
        let fs = DynamicRootFs::new(passthrough::Config::default(), shares);
        fs.init(FsOptions::empty()).unwrap();
        fs
    }

    fn errno(result: io::Result<impl Sized>) -> i32 {
        result.err().unwrap().raw_os_error().unwrap()
    }

    fn lookup(fs: &DynamicRootFs, parent: Inode, name: &str) -> io::Result<Entry> {
        fs.lookup(ctx(), parent, &CString::new(name).unwrap())
    }

    fn read_file(fs: &DynamicRootFs, inode: Inode, handle: Handle) -> Vec<u8> {
        let mut data = Vec::new();
        fs.read(ctx(), inode, handle, ReadInto(&mut data), 4096, 0, None, 0)
            .unwrap();
        data
    }

    fn root_names(fs: &DynamicRootFs, plus: bool) -> Vec<String> {
        let (handle, _) = fs.opendir(ctx(), fuse::ROOT_ID, 0).unwrap();
        let handle = handle.unwrap();
        let mut names = Vec::new();
        let mut add = |entry: DirEntry| {
            names.push(String::from_utf8(entry.name.to_vec()).unwrap());
            Ok(1)
        };
        if plus {
            fs.readdirplus(ctx(), fuse::ROOT_ID, handle, 4096, 0, |dir_entry, entry| {
                assert_eq!(entry.inode, 0);
                add(dir_entry)
            })
            .unwrap();
        } else {
            fs.readdir(ctx(), fuse::ROOT_ID, handle, 4096, 0, add)
                .unwrap();
        }
        fs.releasedir(ctx(), fuse::ROOT_ID, 0, handle).unwrap();
        names
    }

    struct ReadInto<'a>(&'a mut Vec<u8>);

    impl io::Write for ReadInto<'_> {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl ZeroCopyWriter for ReadInto<'_> {
        fn write_from(&mut self, f: &File, count: usize, off: u64) -> io::Result<usize> {
            use std::os::unix::fs::FileExt;
            let mut buf = vec![0; count.min(4096)];
            let n = f.read_at(&mut buf, off)?;
            self.0.extend_from_slice(&buf[..n]);
            Ok(n)
        }
    }

    #[test]
    fn entries_added_after_mount_are_found_and_read() {
        let a = TestDir::new("a");
        let shares = DynamicShares::new();
        let fs = mounted(&shares);
        assert_eq!(
            errno(lookup(&fs, fuse::ROOT_ID, "a")),
            linux_errno::enoent().raw_os_error().unwrap()
        );

        shares.add("a", a.path(), false).unwrap();
        let dir = lookup(&fs, fuse::ROOT_ID, "a").unwrap();
        assert_eq!(dir.entry_timeout, Duration::ZERO);
        assert_eq!(dir.attr_timeout, Duration::ZERO);
        assert_eq!(dir.attr.st_mode & libc::S_IFMT, libc::S_IFDIR);

        let file = lookup(&fs, dir.inode, "file").unwrap();
        let (handle, _) = fs
            .open(ctx(), file.inode, false, libc::O_RDONLY as u32)
            .unwrap();
        assert_eq!(read_file(&fs, file.inode, handle.unwrap()), b"a");
    }

    #[test]
    fn entries_added_before_mount_are_served() {
        let a = TestDir::new("before");
        let shares = DynamicShares::new();
        shares.add("a", a.path(), false).unwrap();
        let fs = mounted(&shares);
        let dir = lookup(&fs, fuse::ROOT_ID, "a").unwrap();
        lookup(&fs, dir.inode, "file").unwrap();
    }

    #[test]
    fn entries_that_cant_be_served_are_dropped() {
        let (gone, kept) = (TestDir::new("gone"), TestDir::new("kept"));
        let shares = DynamicShares::new();
        shares.add("gone", gone.path(), false).unwrap();
        shares.add("kept", kept.path(), false).unwrap();
        fs::remove_dir_all(&gone.0).unwrap();

        let fs = mounted(&shares);
        assert_eq!(root_names(&fs, false), [".", "..", "kept"]);
        assert_eq!(
            errno(lookup(&fs, fuse::ROOT_ID, "gone")),
            linux_errno::enoent().raw_os_error().unwrap()
        );
        shares.add("gone", kept.path(), false).unwrap();
        lookup(&fs, fuse::ROOT_ID, "gone").unwrap();
    }

    #[test]
    fn the_root_lists_the_entries_there_are_when_it_is_opened() {
        let (a, b) = (TestDir::new("list-a"), TestDir::new("list-b"));
        let shares = DynamicShares::new();
        let fs = mounted(&shares);
        assert_eq!(root_names(&fs, false), [".", ".."]);

        shares.add("b", b.path(), false).unwrap();
        shares.add("a", a.path(), false).unwrap();
        for plus in [false, true] {
            assert_eq!(root_names(&fs, plus), [".", "..", "a", "b"]);
        }

        let (handle, _) = fs.opendir(ctx(), fuse::ROOT_ID, 0).unwrap();
        shares.remove("a").unwrap();
        let mut names = Vec::new();
        fs.readdir(ctx(), fuse::ROOT_ID, handle.unwrap(), 4096, 2, |entry| {
            names.push(entry.name.to_vec());
            Ok(1)
        })
        .unwrap();
        assert_eq!(names, [b"a".to_vec(), b"b".to_vec()]);
        assert_eq!(root_names(&fs, true), [".", "..", "b"]);
    }

    #[test]
    fn removed_entries_are_hidden_but_open_files_keep_working() {
        let a = TestDir::new("removed");
        let shares = DynamicShares::new();
        let fs = mounted(&shares);
        shares.add("a", a.path(), false).unwrap();
        let dir = lookup(&fs, fuse::ROOT_ID, "a").unwrap();
        let file = lookup(&fs, dir.inode, "file").unwrap();
        let (handle, _) = fs
            .open(ctx(), file.inode, false, libc::O_RDONLY as u32)
            .unwrap();

        shares.remove("a").unwrap();
        assert_eq!(
            errno(lookup(&fs, fuse::ROOT_ID, "a")),
            linux_errno::enoent().raw_os_error().unwrap()
        );
        assert_eq!(read_file(&fs, file.inode, handle.unwrap()), b"removed");
        lookup(&fs, dir.inode, "file").unwrap();
        assert_eq!(
            errno(shares.remove("a")),
            linux_errno::enoent().raw_os_error().unwrap()
        );

        shares.add("a", a.path(), false).unwrap();
        let again = lookup(&fs, fuse::ROOT_ID, "a").unwrap();
        assert_ne!(again.inode, dir.inode);
    }

    #[test]
    fn a_removed_entry_is_reached_only_through_a_root_the_guest_holds() {
        let dirs = ["never", "forgotten", "held"].map(|name| (name, TestDir::new(name)));
        let shares = DynamicShares::new();
        let fs = mounted(&shares);
        for (name, dir) in &dirs {
            shares.add(name, dir.path(), false).unwrap();
        }
        let ebadf = linux_errno::ebadf().raw_os_error().unwrap();
        let refused = |inode| {
            assert_eq!(errno(lookup(&fs, inode, "file")), ebadf);
            assert_eq!(errno(fs.getattr(ctx(), inode, None)), ebadf);
            assert_eq!(errno(fs.opendir(ctx(), inode, 0)), ebadf);
        };

        // A guest can name the root of an entry without looking it up.
        let never = share_root(1);
        fs.getattr(ctx(), never, None).unwrap();
        let forgotten = lookup(&fs, fuse::ROOT_ID, "forgotten").unwrap().inode;
        fs.forget(ctx(), forgotten, 1);
        let held = lookup(&fs, fuse::ROOT_ID, "held").unwrap().inode;
        assert_eq!(lookup(&fs, held, ".").unwrap().inode, held);
        for (name, _) in &dirs {
            shares.remove(name).unwrap();
        }
        refused(never);
        refused(forgotten);

        lookup(&fs, held, "file").unwrap();
        fs.getattr(ctx(), held, None).unwrap();
        let (handle, _) = fs.opendir(ctx(), held, 0).unwrap();
        fs.releasedir(ctx(), held, 0, handle.unwrap()).unwrap();
        fs.batch_forget(ctx(), vec![(held, 1)]);
        fs.getattr(ctx(), held, None).unwrap();
        fs.forget(ctx(), held, 1);
        refused(held);
    }

    #[test]
    fn shares_have_node_ids_of_their_own() {
        let (a, b) = (TestDir::new("ids-a"), TestDir::new("ids-b"));
        let shares = DynamicShares::new();
        let fs = mounted(&shares);
        shares.add("a", a.path(), false).unwrap();
        shares.add("b", b.path(), false).unwrap();
        let dir_a = lookup(&fs, fuse::ROOT_ID, "a").unwrap();
        let dir_b = lookup(&fs, fuse::ROOT_ID, "b").unwrap();
        let file_a = lookup(&fs, dir_a.inode, "file").unwrap();
        let file_b = lookup(&fs, dir_b.inode, "file").unwrap();

        let ids = [dir_a.inode, dir_b.inode, file_a.inode, file_b.inode];
        for (i, id) in ids.iter().enumerate() {
            assert!(!ids[i + 1..].contains(id));
            assert_ne!(*id, fuse::ROOT_ID);
        }
        assert_eq!(dir_a.inode >> SHARE_SHIFT, file_a.inode >> SHARE_SHIFT);
        assert_ne!(dir_a.inode >> SHARE_SHIFT, dir_b.inode >> SHARE_SHIFT);

        let parent = lookup(&fs, dir_a.inode, ".").unwrap();
        assert_eq!(parent.inode, dir_a.inode);

        assert_eq!(
            errno(fs.link(ctx(), file_a.inode, dir_b.inode, c"link")),
            linux_errno::exdev().raw_os_error().unwrap()
        );
        assert_eq!(
            errno(fs.rename(ctx(), dir_a.inode, c"file", dir_b.inode, c"moved", 0)),
            linux_errno::exdev().raw_os_error().unwrap()
        );
        assert_eq!(
            errno(fs.getattr(ctx(), (1 << SHARE_SHIFT) * 7 + 2, None)),
            linux_errno::ebadf().raw_os_error().unwrap()
        );
    }

    #[test]
    fn read_only_entries_refuse_writes() {
        let (a, b) = (TestDir::new("ro"), TestDir::new("rw"));
        let shares = DynamicShares::new();
        let fs = mounted(&shares);
        shares.add("ro", a.path(), true).unwrap();
        shares.add("rw", b.path(), false).unwrap();
        let erofs = linux_errno::erofs().raw_os_error().unwrap();

        let ro = lookup(&fs, fuse::ROOT_ID, "ro").unwrap();
        let file = lookup(&fs, ro.inode, "file").unwrap();
        assert_eq!(
            errno(fs.open(ctx(), file.inode, false, libc::O_WRONLY as u32)),
            erofs
        );
        assert_eq!(
            errno(fs.mkdir(ctx(), ro.inode, c"dir", 0o755, 0, Extensions::default())),
            erofs
        );
        fs.open(ctx(), file.inode, false, libc::O_RDONLY as u32)
            .unwrap();

        let rw = lookup(&fs, fuse::ROOT_ID, "rw").unwrap();
        fs.mkdir(ctx(), rw.inode, c"dir", 0o755, 0, Extensions::default())
            .unwrap();
        assert!(b.0.join("dir").is_dir());
    }

    #[test]
    fn the_root_itself_cannot_be_changed() {
        let a = TestDir::new("root");
        let shares = DynamicShares::new();
        let fs = mounted(&shares);
        shares.add("a", a.path(), false).unwrap();
        let eacces = linux_errno::eacces().raw_os_error().unwrap();
        assert_eq!(
            errno(fs.mkdir(ctx(), fuse::ROOT_ID, c"b", 0o755, 0, Extensions::default())),
            eacces
        );
        assert_eq!(errno(fs.unlink(ctx(), fuse::ROOT_ID, c"a")), eacces);
        let dir = lookup(&fs, fuse::ROOT_ID, "a").unwrap();
        assert_eq!(
            errno(fs.rename(ctx(), fuse::ROOT_ID, c"a", dir.inode, c"a", 0)),
            eacces
        );
        let (attr, timeout) = fs.getattr(ctx(), fuse::ROOT_ID, None).unwrap();
        assert_eq!(attr.st_mode, libc::S_IFDIR | 0o555);
        assert_eq!(timeout, Duration::ZERO);
    }

    #[test]
    fn names_must_be_one_new_component_and_paths_directories() {
        let a = TestDir::new("names");
        let shares = DynamicShares::new();
        for name in ["", ".", "..", "a/b", "a\0b"] {
            assert_eq!(errno(shares.add(name, a.path(), false)), libc::EINVAL);
        }
        shares.add("a", a.path(), false).unwrap();
        assert_eq!(errno(shares.add("a", a.path(), false)), libc::EEXIST);
        let file = a.0.join("file");
        assert_eq!(
            errno(shares.add("f", file.to_str().unwrap(), false)),
            libc::ENOTDIR
        );
        let missing = a.0.join("missing");
        assert_eq!(
            errno(shares.add("m", missing.to_str().unwrap(), false)),
            libc::ENOENT
        );
    }

    #[test]
    fn policies_see_paths_under_the_entry_name() {
        use std::os::unix::fs::MetadataExt;
        use std::sync::Mutex;

        use crate::virtio::fs::policy::{FsOp, FsPolicy, FsRequest, PolicyFs};

        type Seen = (Option<PathBuf>, Option<(u64, u64)>);
        struct Record(Mutex<Vec<Seen>>);

        impl FsPolicy for Record {
            fn check(&self, request: &FsRequest) -> Result<(), i32> {
                let object = request.object();
                let path = object.path().map(PathBuf::from);
                let node = object.node().map(|node| (node.dev, node.ino));
                let refuse = path.as_ref().is_some_and(|path| path.starts_with("a"))
                    && request.op() != FsOp::Opendir;
                self.0.lock().unwrap().push((path, node));
                if refuse {
                    return Err(libc::EACCES);
                }
                Ok(())
            }
        }

        let (a, b) = (TestDir::new("policy-a"), TestDir::new("policy-b"));
        let shares = DynamicShares::new();
        shares.add("a", a.path(), false).unwrap();
        shares.add("b", b.path(), false).unwrap();
        let record = Arc::new(Record(Mutex::new(Vec::new())));
        let fs = PolicyFs::new(
            DynamicRootFs::new(passthrough::Config::default(), &shares),
            record.clone(),
        );
        fs.init(FsOptions::empty()).unwrap();
        let lookup = |parent, name: &str| {
            fs.lookup(ctx(), parent, &CString::new(name).unwrap())
                .unwrap()
                .inode
        };
        let open = |inode| fs.open(ctx(), inode, false, libc::O_RDONLY as u32);

        let file_a = lookup(lookup(fuse::ROOT_ID, "a"), "file");
        let file_b = lookup(lookup(fuse::ROOT_ID, "b"), "file");
        assert_eq!(
            errno(open(file_a)),
            linux_errno::eacces().raw_os_error().unwrap()
        );
        open(file_b).unwrap();
        fs.opendir(ctx(), fuse::ROOT_ID, 0).unwrap();
        shares.remove("b").unwrap();
        open(file_b).unwrap();

        let meta = fs::metadata(b.0.join("file")).unwrap();
        let seen = record.0.lock().unwrap();
        assert_eq!(seen[0].0, Some(PathBuf::from("a/file")));
        assert_eq!(
            seen[1],
            (
                Some(PathBuf::from("b/file")),
                Some((meta.dev(), meta.ino()))
            )
        );
        assert_eq!(seen[2], (Some(PathBuf::new()), None));
        assert_eq!(seen[3], (None, Some((meta.dev(), meta.ino()))));
    }
}
