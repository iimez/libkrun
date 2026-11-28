// Requests `PolicyFs` doesn't check, and why:
// - `lookup`, `getattr`, `statfs`, `access`: names, attributes and sizes stay visible, so the
//   guest can still walk paths through anything it may not open. `readdirplus` looks entries up
//   inside the passthrough; that is a `lookup` too, and the node IDs it returns are checked like
//   any other when the guest opens, changes or removes them.
// - `read`, `write`, `fallocate`, `lseek`, `copy_file_range`, `flush`, `fsync`, `fsyncdir`,
//   `readdir`, `readdirplus`, `setupmapping`, `ioctl`: they act on a handle that a checked `open`,
//   `opendir` or `create` returned. A policy that narrows applies from the next open; a handle
//   opened before keeps the access it was opened with.
// - `forget`, `release`, `releasedir`, `removemapping`, `init`, `destroy`: housekeeping; refusing
//   them would only leak.

use std::cell::OnceCell;
use std::error::Error;
#[cfg(unix)]
use std::ffi::OsStr;
use std::ffi::{CStr, CString};
use std::fmt;
use std::io;
use std::mem;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

#[cfg(target_os = "macos")]
use crossbeam_channel::Sender;
use utils::eventfd::{EFD_NONBLOCK, EventFd};
#[cfg(target_os = "macos")]
use utils::worker_message::WorkerMessage;

use super::filesystem::{
    Context, DirEntry, Entry, Extensions, FileSystem, FsOptions, GetxattrReply, ListxattrReply,
    OpenOptions, SetattrValid, ZeroCopyReader, ZeroCopyWriter,
};
use super::fuse;
use crate::virtio::bindings;
use crate::virtio::linux_errno;

type Inode = u64;
type Handle = u64;

/// A file or directory on the host, as the host identifies it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HostNode {
    pub dev: u64,
    pub ino: u64,
    /// The mount ID, on hosts that have one (Linux).
    pub mnt_id: Option<u64>,
    /// The file type and permission bits, as in `st_mode`.
    pub mode: u32,
    pub nlink: u64,
}

/// An operation a [`FsPolicy`] is asked about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FsOp {
    /// Open a file. The flags are the guest's open flags.
    Open,
    /// Open a directory to list it. The flags are the guest's open flags.
    Opendir,
    /// Create and open a file. The flags are the guest's open flags.
    Create,
    /// Create a directory. The flags are its mode.
    Mkdir,
    /// Create a device node, FIFO or socket. The flags are its mode.
    Mknod,
    /// Create a symbolic link.
    Symlink,
    /// Give a file another name: the object is the file, the target the new name.
    Link,
    /// Remove a name that isn't a directory.
    Unlink,
    /// Remove a directory.
    Rmdir,
    /// Rename the object to the target. The flags are the `renameat2` flags.
    Rename,
    /// Change attributes: size, mode, owner or times. The flags are the FUSE `FATTR_*` bits of
    /// what is changed.
    Setattr,
    /// Read the target of a symbolic link.
    Readlink,
    /// Read an extended attribute.
    Getxattr,
    /// List the extended attributes.
    Listxattr,
    /// Set an extended attribute. The flags are the `setxattr` flags.
    Setxattr,
    /// Remove an extended attribute.
    Removexattr,
}

/// What a [`FsPolicy`] decides about a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsVerdict {
    /// Let the request through.
    Allow,
    /// Fail the request with this errno, a positive Linux errno value as the guest sees it, such
    /// as `EACCES` (13) or `EROFS` (30). 0 or a negative value fails it with `EACCES`.
    Deny(i32),
    /// Leave the request unanswered until the embedder says it is decided, through
    /// [`FsRequest::deferral`].
    Defer,
}

/// A policy that decides which requests of a virtio-fs share go through.
///
/// `check` runs on the device's worker thread before each request it is asked about, so it
/// should return quickly; while it runs, the share serves nothing else. The embedder may change
/// what it decides at any time, through state of its own: each request is decided when it
/// arrives.
///
/// A policy that needs longer to decide returns [`FsVerdict::Defer`], keeps the request's
/// [`id`](FsRequest::id) and [`deferral`](FsRequest::deferral), and calls
/// [`FsDeferral::decided`] once it knows the answer. The share goes on serving every other
/// request meanwhile. The deferred request is then checked again, with the same id unless the
/// guest changed the request while it waited, and the policy answers it or defers it again.
/// A device holds at most 64 deferred requests; it fails any more at once with `EACCES`, and
/// drops those it holds when the guest resets it.
///
/// Any request may be deferred, but a deferred request costs the guest more than its wait. The
/// guest process that made it can't be killed until it is answered, as the guest sends no
/// interrupt for it. A request that creates, removes, renames or links a name holds the lock of
/// its directory in the guest's kernel, so every other process in the guest that creates a
/// name there, or looks up a name there that it hasn't cached, waits for the answer too; an
/// open of an existing file holds no such lock.
pub trait FsPolicy: Send + Sync {
    /// Decide whether `request` goes through.
    fn check(&self, request: &FsRequest) -> FsVerdict;
}

/// Tells one request apart from every other a [`FsPolicy`] is asked about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FsRequestId(u64);

impl FsRequestId {
    fn next() -> Self {
        // Process-wide, so that a policy shared by devices, or kept across a device reset, never
        // sees an id twice.
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

/// Says that a request a [`FsPolicy`] deferred is decided.
#[derive(Clone)]
pub struct FsDeferral {
    id: FsRequestId,
    deferrals: Weak<Deferrals>,
}

impl FsDeferral {
    /// Check the deferred request again, from any thread. It does nothing for a request that
    /// is no longer held: one already answered, or failed for being over the bound, or dropped
    /// by a reset.
    pub fn decided(&self) {
        if let Some(deferrals) = self.deferrals.upgrade() {
            deferrals.decided.lock().unwrap().push(self.id);
            if let Err(e) = deferrals.event.write(1) {
                error!("virtio-fs policy: can't wake the device for a decided request: {e}");
            }
        }
    }
}

/// The requests a device's policy deferred, shared by the policy's filesystem and the worker
/// that holds them.
pub(crate) struct Deferrals {
    /// Written when a request is decided.
    pub(crate) event: EventFd,
    decided: Mutex<Vec<FsRequestId>>,
    again: Mutex<Option<DeferredRequest>>,
}

impl Deferrals {
    pub(crate) fn new() -> io::Result<Self> {
        Ok(Self {
            event: EventFd::new(EFD_NONBLOCK)?,
            decided: Mutex::new(Vec::new()),
            again: Mutex::new(None),
        })
    }

    /// The requests decided since the last call.
    pub(crate) fn take_decided(&self) -> Vec<FsRequestId> {
        mem::take(&mut self.decided.lock().unwrap())
    }

    /// Check `request` again: the next request the policy is asked about keeps its id if the
    /// guest sent the same request again.
    pub(crate) fn check_again(&self, request: Option<DeferredRequest>) {
        *self.again.lock().unwrap() = request;
    }
}

/// A request a [`FsPolicy`] deferred, as the policy was asked about it.
#[derive(Debug)]
pub struct DeferredRequest {
    id: FsRequestId,
    asked: Asked,
}

impl DeferredRequest {
    pub(crate) fn id(&self) -> FsRequestId {
        self.id
    }
}

impl fmt::Display for DeferredRequest {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "request {} deferred by the policy", self.id.0)
    }
}

impl Error for DeferredRequest {}

/// What a policy sees of a request, to tell whether a deferred request is the same when it is
/// checked again: the guest may have changed it while it waited.
#[derive(Debug, PartialEq, Eq)]
struct Asked {
    op: FsOp,
    uid: u32,
    gid: u32,
    pid: i32,
    flags: u32,
    object: AskedPlace,
    target: Option<AskedPlace>,
    xattr_name: Option<CString>,
}

#[derive(Debug, PartialEq, Eq)]
enum AskedPlace {
    Node(Inode),
    Entry(Inode, CString),
}

impl From<Place<'_>> for AskedPlace {
    fn from(place: Place) -> Self {
        match place {
            Place::Node(inode) => Self::Node(inode),
            Place::Entry { parent, name } => Self::Entry(parent, name.to_owned()),
        }
    }
}

/// What the host knows about the files a node ID or a name stands for.
pub(crate) trait HostNodes {
    /// The host file behind a node ID.
    fn host_node(&self, inode: Inode) -> io::Result<HostNode>;
    /// The host file named `name` in the directory `parent`, not following a symlink, or `None`
    /// if there is none.
    fn host_child(&self, parent: Inode, name: &CStr) -> io::Result<Option<HostNode>>;
    /// The path of a node ID relative to the root of the share, as the host resolves its
    /// descriptor now, or `None` if it is no longer reachable from the root.
    fn host_path(&self, inode: Inode) -> io::Result<Option<PathBuf>>;
}

#[derive(Clone, Copy)]
enum Place<'a> {
    Node(Inode),
    Entry { parent: Inode, name: &'a CStr },
}

/// A file, or a name in a directory, that a request is about. The host is asked about it only
/// when the policy calls a method, at most once per method and request.
pub struct FsObject<'a> {
    nodes: &'a dyn HostNodes,
    place: Place<'a>,
    node: OnceCell<Option<HostNode>>,
    parent: OnceCell<Option<HostNode>>,
    path: OnceCell<Option<PathBuf>>,
}

impl<'a> FsObject<'a> {
    fn new(nodes: &'a dyn HostNodes, place: Place<'a>) -> Self {
        Self {
            nodes,
            place,
            node: OnceCell::new(),
            parent: OnceCell::new(),
            path: OnceCell::new(),
        }
    }

    /// The host file: the one the node ID stands for, or the one at the name now, if any. It
    /// is `None` for a name that is about to be created.
    pub fn node(&self) -> Option<HostNode> {
        *self.node.get_or_init(|| {
            match self.place {
                Place::Node(inode) => self.nodes.host_node(inode).map(Some),
                Place::Entry { parent, name } => self.nodes.host_child(parent, name),
            }
            .unwrap_or_else(|e| {
                debug!("virtio-fs policy: can't identify a file: {e}");
                None
            })
        })
    }

    /// The directory the name is in, when the object is a name.
    pub fn parent(&self) -> Option<HostNode> {
        let Place::Entry { parent, .. } = self.place else {
            return None;
        };
        *self
            .parent
            .get_or_init(|| self.nodes.host_node(parent).ok())
    }

    /// The name in its directory, when the object is a name.
    pub fn name(&self) -> Option<&'a CStr> {
        match self.place {
            Place::Node(_) => None,
            Place::Entry { name, .. } => Some(name),
        }
    }

    /// The path relative to the root of the share, with no leading `/`; the root itself is the
    /// empty path. It is `None` when the host can't tell: the file was removed, moved out of the
    /// share, or the name isn't a single path component.
    pub fn path(&self) -> Option<&Path> {
        self.path
            .get_or_init(|| {
                let path = match self.place {
                    Place::Node(inode) => self.nodes.host_path(inode),
                    Place::Entry { parent, name } => {
                        let name = component(name)?;
                        self.nodes
                            .host_path(parent)
                            .map(|dir| dir.map(|dir| dir.join(name)))
                    }
                };
                path.unwrap_or_else(|e| {
                    debug!("virtio-fs policy: can't find the path of a file: {e}");
                    None
                })
            })
            .as_deref()
    }
}

/// `name` as a path, if it is a single path component.
fn component(name: &CStr) -> Option<&Path> {
    let name = name.to_bytes();
    if matches!(name, b"" | b"." | b"..") || name.contains(&b'/') {
        return None;
    }
    #[cfg(unix)]
    return Some(Path::new(OsStr::from_bytes(name)));
    // A Windows path holds only the names that are UTF-8.
    #[cfg(not(unix))]
    str::from_utf8(name).ok().map(Path::new)
}

/// A request a [`FsPolicy`] is asked about.
pub struct FsRequest<'a> {
    id: FsRequestId,
    op: FsOp,
    ctx: Context,
    flags: u32,
    object: FsObject<'a>,
    target: Option<FsObject<'a>>,
    xattr_name: Option<&'a CStr>,
    deferrals: &'a Arc<Deferrals>,
}

impl<'a> FsRequest<'a> {
    /// Tells the request apart from the others. A deferred request has the same id each time
    /// it is checked again, as long as the guest sends the same request.
    pub fn id(&self) -> FsRequestId {
        self.id
    }

    /// What a policy that defers the request calls once it is decided.
    pub fn deferral(&self) -> FsDeferral {
        FsDeferral {
            id: self.id,
            deferrals: Arc::downgrade(self.deferrals),
        }
    }

    pub fn op(&self) -> FsOp {
        self.op
    }

    /// The user ID of the guest process that made the request.
    pub fn uid(&self) -> u32 {
        self.ctx.uid
    }

    /// The group ID of the guest process that made the request.
    pub fn gid(&self) -> u32 {
        self.ctx.gid
    }

    /// The thread group ID, in the guest, of the process that made the request.
    pub fn pid(&self) -> i32 {
        self.ctx.pid
    }

    /// The flags of the operation, as [`FsOp`] describes for each; 0 for the others.
    pub fn flags(&self) -> u32 {
        self.flags
    }

    /// What the operation acts on: the file for operations on a node, the name for those that
    /// create or remove one, and the source of a rename.
    pub fn object(&self) -> &FsObject<'a> {
        &self.object
    }

    /// The new name of a `Link` or a `Rename`.
    pub fn target(&self) -> Option<&FsObject<'a>> {
        self.target.as_ref()
    }

    /// The name of the extended attribute of a `Getxattr`, `Setxattr` or `Removexattr`.
    pub fn xattr_name(&self) -> Option<&'a CStr> {
        self.xattr_name
    }
}

/// A filesystem whose requests are checked by a [`FsPolicy`] before `inner` serves them.
pub struct PolicyFs<T> {
    inner: T,
    policy: Arc<dyn FsPolicy>,
    deferrals: Arc<Deferrals>,
}

impl<T: FileSystem<Inode = Inode, Handle = Handle> + HostNodes> PolicyFs<T> {
    /// A request `policy` defers fails with a [`DeferredRequest`] error, for the device to hold
    /// and to check again through `deferrals`.
    pub fn new(inner: T, policy: Arc<dyn FsPolicy>, deferrals: Arc<Deferrals>) -> Self {
        Self {
            inner,
            policy,
            deferrals,
        }
    }

    fn check(
        &self,
        ctx: Context,
        op: FsOp,
        flags: u32,
        object: Place,
        target: Option<Place>,
    ) -> io::Result<()> {
        self.ask(ctx, op, flags, object, target, None)
    }

    fn check_xattr(
        &self,
        ctx: Context,
        op: FsOp,
        flags: u32,
        inode: Inode,
        name: Option<&CStr>,
    ) -> io::Result<()> {
        self.ask(ctx, op, flags, Place::Node(inode), None, name)
    }

    fn ask(
        &self,
        ctx: Context,
        op: FsOp,
        flags: u32,
        object: Place,
        target: Option<Place>,
        xattr_name: Option<&CStr>,
    ) -> io::Result<()> {
        let asked = || Asked {
            op,
            uid: ctx.uid,
            gid: ctx.gid,
            pid: ctx.pid,
            flags,
            object: object.into(),
            target: target.map(Into::into),
            xattr_name: xattr_name.map(CStr::to_owned),
        };
        let id = match self.deferrals.again.lock().unwrap().take() {
            Some(deferred) if deferred.asked == asked() => deferred.id,
            _ => FsRequestId::next(),
        };
        let request = FsRequest {
            id,
            op,
            ctx,
            flags,
            object: FsObject::new(&self.inner, object),
            target: target.map(|place| FsObject::new(&self.inner, place)),
            xattr_name,
            deferrals: &self.deferrals,
        };
        match self.policy.check(&request) {
            FsVerdict::Allow => Ok(()),
            FsVerdict::Deny(errno) if errno > 0 => Err(io::Error::from_raw_os_error(errno)),
            FsVerdict::Deny(_) => Err(linux_errno::eacces()),
            FsVerdict::Defer => Err(io::Error::other(DeferredRequest { id, asked: asked() })),
        }
    }
}

impl<T: FileSystem<Inode = Inode, Handle = Handle> + HostNodes> FileSystem for PolicyFs<T> {
    type Inode = Inode;
    type Handle = Handle;

    fn init(&self, capable: FsOptions) -> io::Result<FsOptions> {
        self.inner.init(capable)
    }

    fn destroy(&self) {
        self.inner.destroy()
    }

    fn lookup(&self, ctx: Context, parent: Inode, name: &CStr) -> io::Result<Entry> {
        self.inner.lookup(ctx, parent, name)
    }

    fn forget(&self, ctx: Context, inode: Inode, count: u64) {
        self.inner.forget(ctx, inode, count)
    }

    fn batch_forget(&self, ctx: Context, requests: Vec<(Inode, u64)>) {
        self.inner.batch_forget(ctx, requests)
    }

    fn getattr(
        &self,
        ctx: Context,
        inode: Inode,
        handle: Option<Handle>,
    ) -> io::Result<(bindings::stat64, Duration)> {
        self.inner.getattr(ctx, inode, handle)
    }

    fn setattr(
        &self,
        ctx: Context,
        inode: Inode,
        attr: bindings::stat64,
        handle: Option<Handle>,
        valid: SetattrValid,
    ) -> io::Result<(bindings::stat64, Duration)> {
        self.check(ctx, FsOp::Setattr, valid.bits(), Place::Node(inode), None)?;
        self.inner.setattr(ctx, inode, attr, handle, valid)
    }

    fn readlink(&self, ctx: Context, inode: Inode) -> io::Result<Vec<u8>> {
        self.check(ctx, FsOp::Readlink, 0, Place::Node(inode), None)?;
        self.inner.readlink(ctx, inode)
    }

    fn symlink(
        &self,
        ctx: Context,
        linkname: &CStr,
        parent: Inode,
        name: &CStr,
        extensions: Extensions,
    ) -> io::Result<Entry> {
        self.check(ctx, FsOp::Symlink, 0, Place::Entry { parent, name }, None)?;
        self.inner.symlink(ctx, linkname, parent, name, extensions)
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
        self.check(ctx, FsOp::Mknod, mode, Place::Entry { parent, name }, None)?;
        self.inner
            .mknod(ctx, parent, name, mode, rdev, umask, extensions)
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
        self.check(ctx, FsOp::Mkdir, mode, Place::Entry { parent, name }, None)?;
        self.inner.mkdir(ctx, parent, name, mode, umask, extensions)
    }

    fn unlink(&self, ctx: Context, parent: Inode, name: &CStr) -> io::Result<()> {
        self.check(ctx, FsOp::Unlink, 0, Place::Entry { parent, name }, None)?;
        self.inner.unlink(ctx, parent, name)
    }

    fn rmdir(&self, ctx: Context, parent: Inode, name: &CStr) -> io::Result<()> {
        self.check(ctx, FsOp::Rmdir, 0, Place::Entry { parent, name }, None)?;
        self.inner.rmdir(ctx, parent, name)
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
        self.check(
            ctx,
            FsOp::Rename,
            flags,
            Place::Entry {
                parent: olddir,
                name: oldname,
            },
            Some(Place::Entry {
                parent: newdir,
                name: newname,
            }),
        )?;
        self.inner
            .rename(ctx, olddir, oldname, newdir, newname, flags)
    }

    fn link(
        &self,
        ctx: Context,
        inode: Inode,
        newparent: Inode,
        newname: &CStr,
    ) -> io::Result<Entry> {
        self.check(
            ctx,
            FsOp::Link,
            0,
            Place::Node(inode),
            Some(Place::Entry {
                parent: newparent,
                name: newname,
            }),
        )?;
        self.inner.link(ctx, inode, newparent, newname)
    }

    fn open(
        &self,
        ctx: Context,
        inode: Inode,
        kill_priv: bool,
        flags: u32,
    ) -> io::Result<(Option<Handle>, OpenOptions)> {
        self.check(ctx, FsOp::Open, flags, Place::Node(inode), None)?;
        self.inner.open(ctx, inode, kill_priv, flags)
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
        self.check(
            ctx,
            FsOp::Create,
            flags,
            Place::Entry { parent, name },
            None,
        )?;
        self.inner
            .create(ctx, parent, name, mode, kill_priv, flags, umask, extensions)
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
        self.inner
            .read(ctx, inode, handle, w, size, offset, lock_owner, flags)
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
        self.inner.write(
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
        )
    }

    fn flush(&self, ctx: Context, inode: Inode, handle: Handle, lock_owner: u64) -> io::Result<()> {
        self.inner.flush(ctx, inode, handle, lock_owner)
    }

    fn fsync(&self, ctx: Context, inode: Inode, datasync: bool, handle: Handle) -> io::Result<()> {
        self.inner.fsync(ctx, inode, datasync, handle)
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
        self.inner
            .fallocate(ctx, inode, handle, mode, offset, length)
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
        self.inner
            .release(ctx, inode, flags, handle, flush, flock_release, lock_owner)
    }

    fn statfs(&self, ctx: Context, inode: Inode) -> io::Result<bindings::statvfs64> {
        self.inner.statfs(ctx, inode)
    }

    fn setxattr(
        &self,
        ctx: Context,
        inode: Inode,
        name: &CStr,
        value: &[u8],
        flags: u32,
    ) -> io::Result<()> {
        self.check_xattr(ctx, FsOp::Setxattr, flags, inode, Some(name))?;
        self.inner.setxattr(ctx, inode, name, value, flags)
    }

    fn getxattr(
        &self,
        ctx: Context,
        inode: Inode,
        name: &CStr,
        size: u32,
    ) -> io::Result<GetxattrReply> {
        self.check_xattr(ctx, FsOp::Getxattr, 0, inode, Some(name))?;
        self.inner.getxattr(ctx, inode, name, size)
    }

    fn listxattr(&self, ctx: Context, inode: Inode, size: u32) -> io::Result<ListxattrReply> {
        self.check_xattr(ctx, FsOp::Listxattr, 0, inode, None)?;
        self.inner.listxattr(ctx, inode, size)
    }

    fn removexattr(&self, ctx: Context, inode: Inode, name: &CStr) -> io::Result<()> {
        self.check_xattr(ctx, FsOp::Removexattr, 0, inode, Some(name))?;
        self.inner.removexattr(ctx, inode, name)
    }

    fn opendir(
        &self,
        ctx: Context,
        inode: Inode,
        flags: u32,
    ) -> io::Result<(Option<Handle>, OpenOptions)> {
        self.check(ctx, FsOp::Opendir, flags, Place::Node(inode), None)?;
        self.inner.opendir(ctx, inode, flags)
    }

    fn readdir<F>(
        &self,
        ctx: Context,
        inode: Inode,
        handle: Handle,
        size: u32,
        offset: u64,
        add_entry: F,
    ) -> io::Result<()>
    where
        F: FnMut(DirEntry) -> io::Result<usize>,
    {
        self.inner
            .readdir(ctx, inode, handle, size, offset, add_entry)
    }

    fn readdirplus<F>(
        &self,
        ctx: Context,
        inode: Inode,
        handle: Handle,
        size: u32,
        offset: u64,
        add_entry: F,
    ) -> io::Result<()>
    where
        F: FnMut(DirEntry, Entry) -> io::Result<usize>,
    {
        self.inner
            .readdirplus(ctx, inode, handle, size, offset, add_entry)
    }

    fn fsyncdir(
        &self,
        ctx: Context,
        inode: Inode,
        datasync: bool,
        handle: Handle,
    ) -> io::Result<()> {
        self.inner.fsyncdir(ctx, inode, datasync, handle)
    }

    fn releasedir(&self, ctx: Context, inode: Inode, flags: u32, handle: Handle) -> io::Result<()> {
        self.inner.releasedir(ctx, inode, flags, handle)
    }

    fn access(&self, ctx: Context, inode: Inode, mask: u32) -> io::Result<()> {
        self.inner.access(ctx, inode, mask)
    }

    fn lseek(
        &self,
        ctx: Context,
        inode: Inode,
        handle: Handle,
        offset: u64,
        whence: u32,
    ) -> io::Result<u64> {
        self.inner.lseek(ctx, inode, handle, offset, whence)
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
        self.inner.copyfilerange(
            ctx, inode_in, handle_in, offset_in, inode_out, handle_out, offset_out, len, flags,
        )
    }

    fn setupmapping(
        &self,
        ctx: Context,
        inode: Inode,
        handle: Handle,
        foffset: u64,
        len: u64,
        flags: u64,
        moffset: u64,
        host_shm_base: u64,
        shm_size: u64,
        #[cfg(target_os = "macos")] map_sender: &Option<Sender<WorkerMessage>>,
    ) -> io::Result<()> {
        self.inner.setupmapping(
            ctx,
            inode,
            handle,
            foffset,
            len,
            flags,
            moffset,
            host_shm_base,
            shm_size,
            #[cfg(target_os = "macos")]
            map_sender,
        )
    }

    fn removemapping(
        &self,
        ctx: Context,
        requests: Vec<fuse::RemovemappingOne>,
        host_shm_base: u64,
        shm_size: u64,
        #[cfg(target_os = "macos")] map_sender: &Option<Sender<WorkerMessage>>,
    ) -> io::Result<()> {
        self.inner.removemapping(
            ctx,
            requests,
            host_shm_base,
            shm_size,
            #[cfg(target_os = "macos")]
            map_sender,
        )
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
        self.inner.ioctl(
            ctx, inode, handle, flags, cmd, arg, in_size, out_size, exit_code,
        )
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::ffi::CString;
    use std::fs::{self, File};
    use std::os::unix::fs::{FileExt, MetadataExt};
    use std::sync::RwLock;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;
    use crate::virtio::fs::inode_alloc::InodeAllocator;
    use crate::virtio::fs::passthrough::{self, PassthroughFs};

    // Distinct from anything the passthrough returns, so a refusal is told apart.
    const REFUSED: i32 = 125;

    type Rule = Box<dyn Fn(&FsRequest) -> FsVerdict + Send + Sync>;

    struct Rules(RwLock<Rule>);

    impl FsPolicy for Rules {
        fn check(&self, request: &FsRequest) -> FsVerdict {
            (self.0.read().unwrap())(request)
        }
    }

    struct Share {
        dir: PathBuf,
        rules: Arc<Rules>,
        deferrals: Arc<Deferrals>,
        fs: PolicyFs<PassthroughFs>,
    }

    impl Share {
        fn new(name: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("libkrun-policy-{}-{name}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(dir.join("dir")).unwrap();
            fs::write(dir.join("file"), b"contents").unwrap();
            std::os::unix::fs::symlink("file", dir.join("symlink")).unwrap();
            let cfg = passthrough::Config {
                root_dir: dir.to_str().unwrap().to_owned(),
                ..Default::default()
            };
            let inner = PassthroughFs::new(cfg, Arc::new(InodeAllocator::new())).unwrap();
            let rules = Arc::new(Rules(RwLock::new(Box::new(|_| FsVerdict::Allow))));
            let deferrals = Arc::new(Deferrals::new().unwrap());
            let fs = PolicyFs::new(inner, rules.clone(), deferrals.clone());
            fs.init(FsOptions::empty()).unwrap();
            Share {
                dir,
                rules,
                deferrals,
                fs,
            }
        }

        fn rule(&self, rule: impl Fn(&FsRequest) -> FsVerdict + Send + Sync + 'static) {
            *self.rules.0.write().unwrap() = Box::new(rule);
        }

        fn inode(&self, name: &str) -> Inode {
            self.lookup(fuse::ROOT_ID, name).unwrap().inode
        }

        fn lookup(&self, parent: Inode, name: &str) -> io::Result<Entry> {
            self.fs.lookup(ctx(), parent, &CString::new(name).unwrap())
        }

        fn open(&self, name: &str, flags: i32) -> io::Result<Handle> {
            let inode = self.inode(name);
            Ok(self.fs.open(ctx(), inode, false, flags as u32)?.0.unwrap())
        }

        fn create(&self, name: &str) -> io::Result<()> {
            let name = CString::new(name).unwrap();
            self.fs
                .create(
                    ctx(),
                    fuse::ROOT_ID,
                    &name,
                    0o644,
                    false,
                    libc::O_WRONLY as u32,
                    0,
                    Extensions::default(),
                )
                .map(drop)
        }

        fn rename(&self, from: &str, to: &str) -> io::Result<()> {
            let (from, to) = (CString::new(from).unwrap(), CString::new(to).unwrap());
            self.fs
                .rename(ctx(), fuse::ROOT_ID, &from, fuse::ROOT_ID, &to, 0)
        }
    }

    impl Drop for Share {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
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

    fn refused(result: io::Result<impl Sized>) -> bool {
        matches!(result, Err(e) if e.raw_os_error() == Some(REFUSED))
    }

    fn identity(node: HostNode) -> (u64, u64) {
        (node.dev, node.ino)
    }

    struct Data<'a>(&'a [u8]);

    impl io::Read for Data<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.0.read(buf)
        }
    }

    impl ZeroCopyReader for Data<'_> {
        fn read_to(&mut self, f: &File, count: usize, off: u64) -> io::Result<usize> {
            let n = count.min(self.0.len());
            let written = f.write_at(&self.0[..n], off)?;
            self.0 = &self.0[written..];
            Ok(written)
        }
    }

    fn run(share: &Share, op: FsOp) -> io::Result<()> {
        let c = ctx();
        let root = fuse::ROOT_ID;
        let file = share.inode("file");
        let ext = Extensions::default;
        match op {
            FsOp::Open => share.open("file", libc::O_RDONLY).map(drop),
            FsOp::Opendir => share
                .fs
                .opendir(c, share.inode("dir"), libc::O_RDONLY as u32)
                .map(drop),
            FsOp::Create => share.create("new"),
            FsOp::Mkdir => share.fs.mkdir(c, root, c"new", 0o755, 0, ext()).map(drop),
            FsOp::Mknod => share
                .fs
                .mknod(c, root, c"fifo", (libc::S_IFIFO | 0o644) as _, 0, 0, ext())
                .map(drop),
            FsOp::Symlink => share.fs.symlink(c, c"file", root, c"new", ext()).map(drop),
            FsOp::Link => share.fs.link(c, file, root, c"new").map(drop),
            FsOp::Unlink => share.fs.unlink(c, root, c"file"),
            FsOp::Rmdir => share.fs.rmdir(c, root, c"dir"),
            FsOp::Rename => share.rename("file", "new"),
            FsOp::Setattr => {
                // SAFETY: stat64 is plain data, for which all-zero bytes are a valid value.
                let mut attr: bindings::stat64 = unsafe { std::mem::zeroed() };
                attr.st_mode = 0o600;
                share
                    .fs
                    .setattr(c, file, attr, None, SetattrValid::MODE)
                    .map(drop)
            }
            FsOp::Readlink => share.fs.readlink(c, share.inode("symlink")).map(drop),
            FsOp::Getxattr => share.fs.getxattr(c, file, c"user.a", 0).map(drop),
            FsOp::Listxattr => share.fs.listxattr(c, file, 0).map(drop),
            FsOp::Setxattr => share.fs.setxattr(c, file, c"user.a", b"1", 0),
            FsOp::Removexattr => share.fs.removexattr(c, file, c"user.a"),
        }
    }

    const CHECKED: [FsOp; 16] = [
        FsOp::Open,
        FsOp::Opendir,
        FsOp::Create,
        FsOp::Mkdir,
        FsOp::Mknod,
        FsOp::Symlink,
        FsOp::Link,
        FsOp::Unlink,
        FsOp::Rmdir,
        FsOp::Rename,
        FsOp::Setattr,
        FsOp::Readlink,
        FsOp::Getxattr,
        FsOp::Listxattr,
        FsOp::Setxattr,
        FsOp::Removexattr,
    ];

    fn is_xattr(op: FsOp) -> bool {
        matches!(
            op,
            FsOp::Getxattr | FsOp::Listxattr | FsOp::Setxattr | FsOp::Removexattr
        )
    }

    #[test]
    fn every_checked_operation_is_refused_and_allowed() {
        for op in CHECKED {
            let share = Share::new(&format!("refused-{op:?}"));
            share.rule(move |request| {
                if request.op() == op {
                    FsVerdict::Deny(REFUSED)
                } else {
                    FsVerdict::Allow
                }
            });
            assert!(refused(run(&share, op)), "{op:?} went through");
            assert_eq!(fs::read(share.dir.join("file")).unwrap(), b"contents");
            assert!(share.dir.join("dir").is_dir());
            assert!(!share.dir.join("new").exists());

            let share = Share::new(&format!("allowed-{op:?}"));
            let result = run(&share, op);
            // Extended attributes are off by default, so these fail, but not as refused.
            if is_xattr(op) {
                assert!(!refused(result), "{op:?} was refused");
            } else {
                result.unwrap_or_else(|e| panic!("{op:?}: {e}"));
            }
        }
    }

    #[test]
    fn requests_name_their_objects() {
        let share = Share::new("objects");
        let seen = Arc::new(RwLock::new(Vec::new()));
        let record = seen.clone();
        share.rule(move |request| {
            let object = request.object();
            let target = request.target();
            record.write().unwrap().push((
                request.op(),
                request.flags(),
                object.path().map(Path::to_path_buf),
                object.name().map(CStr::to_owned),
                object.node().map(identity),
                object.parent().map(identity),
                target.and_then(|t| t.path().map(Path::to_path_buf)),
                target.and_then(|t| t.node().map(identity)),
            ));
            FsVerdict::Allow
        });
        let file = fs::metadata(share.dir.join("file")).unwrap();
        let root = fs::metadata(&share.dir).unwrap();
        let file_id = Some((file.dev(), file.ino()));
        let root_id = Some((root.dev(), root.ino()));

        share.open("file", libc::O_RDWR).unwrap();
        share.create("new").unwrap();
        share.rename("new", "file").unwrap();
        share
            .fs
            .opendir(ctx(), fuse::ROOT_ID, libc::O_RDONLY as u32)
            .unwrap();

        let seen = seen.read().unwrap();
        assert_eq!(seen.len(), 4);
        assert_eq!(
            seen[0],
            (
                FsOp::Open,
                libc::O_RDWR as u32,
                Some("file".into()),
                None,
                file_id,
                None,
                None,
                None
            )
        );
        assert_eq!(seen[1].0, FsOp::Create);
        assert_eq!(seen[1].2, Some("new".into()));
        assert_eq!(seen[1].3.as_deref(), Some(c"new"));
        assert_eq!(seen[1].4, None, "a name about to be created has no file");
        assert_eq!(seen[1].5, root_id);
        assert_eq!(seen[2].0, FsOp::Rename);
        assert_eq!(seen[2].2, Some("new".into()));
        assert_eq!(seen[2].6, Some("file".into()));
        assert_eq!(seen[2].7, file_id, "the file a rename replaces");
        assert_eq!(seen[3].2, Some("".into()));
        assert_eq!(seen[3].4, root_id);
    }

    #[test]
    fn paths_are_where_the_host_has_the_file_now() {
        let share = Share::new("paths");
        let paths = Arc::new(RwLock::new(Vec::new()));
        let record = paths.clone();
        share.rule(move |request| {
            record
                .write()
                .unwrap()
                .push(request.object().path().map(Path::to_path_buf));
            FsVerdict::Allow
        });
        let file = share.inode("file");
        let dir = share.inode("dir");
        fs::rename(share.dir.join("file"), share.dir.join("dir/moved")).unwrap();
        share
            .fs
            .open(ctx(), file, false, libc::O_RDONLY as u32)
            .unwrap();
        share
            .fs
            .mkdir(ctx(), dir, c"sub", 0o755, 0, Extensions::default())
            .unwrap();
        fs::remove_file(share.dir.join("dir/moved")).unwrap();
        let _ = share.fs.open(ctx(), file, false, libc::O_RDONLY as u32);
        assert_eq!(
            *paths.read().unwrap(),
            [Some("dir/moved".into()), Some("dir/sub".into()), None]
        );
    }

    // Linux's readlink of /proc/self/fd says " (deleted)"; macOS's F_GETPATH has no such suffix.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_removed_name_is_no_path_even_while_the_file_has_others() {
        let share = Share::new("removed-name");
        let paths = Arc::new(RwLock::new(Vec::new()));
        let record = paths.clone();
        share.rule(move |request| {
            record
                .write()
                .unwrap()
                .push(request.object().path().map(Path::to_path_buf));
            FsVerdict::Allow
        });
        fs::hard_link(share.dir.join("file"), share.dir.join("other")).unwrap();
        fs::write(share.dir.join("real (deleted)"), b"").unwrap();
        let file = share.inode("file");
        let real = share.inode("real (deleted)");
        fs::remove_file(share.dir.join("file")).unwrap();
        let _ = share.fs.open(ctx(), file, false, libc::O_RDONLY as u32);
        share
            .fs
            .open(ctx(), real, false, libc::O_RDONLY as u32)
            .unwrap();
        assert_eq!(
            *paths.read().unwrap(),
            [None, Some("real (deleted)".into())]
        );
    }

    #[test]
    fn lookups_and_attributes_are_not_checked() {
        let share = Share::new("unchecked");
        share.rule(|_| FsVerdict::Deny(REFUSED));
        let entry = share.lookup(fuse::ROOT_ID, "file").unwrap();
        assert_eq!(entry.attr.st_size, 8);
        share.fs.getattr(ctx(), entry.inode, None).unwrap();
        share.fs.statfs(ctx(), entry.inode).unwrap();
        assert!(refused(share.open("file", libc::O_RDONLY)));
    }

    #[test]
    fn hard_links_of_a_protected_file_are_the_same_file() {
        let share = Share::new("links");
        fs::hard_link(share.dir.join("file"), share.dir.join("alias")).unwrap();
        let meta = fs::metadata(share.dir.join("file")).unwrap();
        let protected = (meta.dev(), meta.ino());
        share.rule(move |request| {
            let objects = [Some(request.object()), request.target()];
            if objects
                .into_iter()
                .flatten()
                .any(|object| object.node().map(identity) == Some(protected))
            {
                return FsVerdict::Deny(REFUSED);
            }
            FsVerdict::Allow
        });

        assert!(refused(share.open("alias", libc::O_RDWR)));
        let alias = share.inode("alias");
        assert!(refused(share.fs.link(
            ctx(),
            alias,
            fuse::ROOT_ID,
            c"another"
        )));
        assert!(refused(share.fs.unlink(ctx(), fuse::ROOT_ID, c"alias")));
        assert!(refused(share.rename("dir", "alias")));
        share.create("other").unwrap();
        assert!(share.dir.join("alias").exists());
    }

    #[test]
    fn readdirplus_doesnt_get_around_the_policy() {
        let share = Share::new("readdirplus");
        let meta = fs::metadata(share.dir.join("file")).unwrap();
        let protected = (meta.dev(), meta.ino());
        share.rule(move |request| {
            if request.op() == FsOp::Open
                && request.object().node().map(identity) == Some(protected)
            {
                return FsVerdict::Deny(REFUSED);
            }
            FsVerdict::Allow
        });

        let (handle, _) = share
            .fs
            .opendir(ctx(), fuse::ROOT_ID, libc::O_RDONLY as u32)
            .unwrap();
        let mut entries = Vec::new();
        share
            .fs
            .readdirplus(
                ctx(),
                fuse::ROOT_ID,
                handle.unwrap(),
                4096,
                0,
                |dir_entry, entry| {
                    entries.push((dir_entry.name.to_vec(), entry.inode));
                    Ok(1)
                },
            )
            .unwrap();
        let inode_of = |name: &[u8]| entries.iter().find(|(n, _)| n == name).unwrap().1;

        let open = |inode| share.fs.open(ctx(), inode, false, libc::O_RDONLY as u32);
        assert!(refused(open(inode_of(b"file"))));
        share
            .fs
            .opendir(ctx(), inode_of(b"dir"), libc::O_RDONLY as u32)
            .unwrap();
    }

    #[test]
    fn a_changed_policy_applies_from_the_next_request() {
        let share = Share::new("changed");
        let read_only = Arc::new(AtomicBool::new(false));
        let flag = read_only.clone();
        share.rule(move |request| {
            let writes = match request.op() {
                FsOp::Open => request.flags() & libc::O_ACCMODE as u32 != libc::O_RDONLY as u32,
                FsOp::Opendir | FsOp::Readlink | FsOp::Getxattr | FsOp::Listxattr => false,
                _ => true,
            };
            if writes && flag.load(Ordering::Relaxed) {
                return FsVerdict::Deny(libc::EROFS);
            }
            FsVerdict::Allow
        });
        fn erofs(result: io::Result<impl Sized>) -> bool {
            matches!(result, Err(e) if e.raw_os_error() == Some(libc::EROFS))
        }
        let file = share.inode("file");
        let write = |handle, data: &[u8]| {
            share.fs.write(
                ctx(),
                file,
                handle,
                Data(data),
                data.len() as u32,
                0,
                None,
                false,
                false,
                0,
            )
        };

        let before = share.open("file", libc::O_WRONLY).unwrap();
        read_only.store(true, Ordering::Relaxed);
        assert!(erofs(share.open("file", libc::O_WRONLY)));
        assert!(erofs(share.create("new")));
        assert!(erofs(share.rename("file", "new")));
        share.open("file", libc::O_RDONLY).unwrap();
        // A handle opened before keeps the access it was opened with.
        write(before, b"CONTENTS").unwrap();
        assert_eq!(fs::read(share.dir.join("file")).unwrap(), b"CONTENTS");

        read_only.store(false, Ordering::Relaxed);
        let after = share.open("file", libc::O_WRONLY).unwrap();
        write(after, b"c").unwrap();
        share.create("new").unwrap();
    }

    #[test]
    fn a_deferred_request_keeps_its_id_only_while_it_is_the_same() {
        let share = Share::new("deferred");
        let ids = Arc::new(RwLock::new(Vec::new()));
        let record = ids.clone();
        share.rule(move |request| {
            record.write().unwrap().push(request.id());
            FsVerdict::Defer
        });
        let dir = share.inode("dir");
        let create = |parent, name: &CStr, flags: i32| {
            let result = share.fs.create(
                ctx(),
                parent,
                name,
                0o644,
                false,
                flags as u32,
                0,
                Extensions::default(),
            );
            result
                .map(drop)
                .unwrap_err()
                .downcast::<DeferredRequest>()
                .unwrap()
        };
        let again = |request: DeferredRequest| share.deferrals.check_again(Some(request));

        let first = create(fuse::ROOT_ID, c"new", libc::O_WRONLY);
        again(first);
        let same = create(fuse::ROOT_ID, c"new", libc::O_WRONLY);
        again(same);
        let renamed = create(fuse::ROOT_ID, c"other", libc::O_WRONLY);
        again(renamed);
        let moved = create(dir, c"other", libc::O_WRONLY);
        again(moved);
        let reflagged = create(dir, c"other", libc::O_RDWR);
        let fresh = create(dir, c"other", libc::O_RDWR);

        let ids = ids.read().unwrap();
        assert_eq!(ids[0], ids[1], "the same request got a new id");
        for i in 2..ids.len() {
            assert!(
                !ids[..i].contains(&ids[i]),
                "request {i} took an earlier id"
            );
        }
        assert_eq!(fresh.id(), ids[5]);
        assert_eq!(reflagged.id(), ids[4]);
        assert!(!share.dir.join("new").exists());
    }
}
