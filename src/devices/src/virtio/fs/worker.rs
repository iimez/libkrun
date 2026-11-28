#[cfg(target_os = "macos")]
use crossbeam_channel::Sender;
#[cfg(target_os = "macos")]
use utils::worker_message::WorkerMessage;

#[cfg(not(unix))]
use std::convert::Infallible;
use std::io;
#[cfg(unix)]
use std::os::fd::AsRawFd;
use std::sync::Arc;
use std::sync::atomic::AtomicI32;
use std::thread;
#[cfg(windows)]
use utils::windows::AsRawFd;

use utils::epoll::{ControlOperation, Epoll, EpollEvent, EventSet};
use utils::eventfd::EventFd;
use vm_memory::GuestMemoryMmap;

use super::super::{DescriptorChain, FsError, Queue};
use super::augment_fs::AugmentFs;
use super::defs::{HPQ_INDEX, REQ_INDEX};
use super::descriptor_utils::{Reader, Writer};
#[cfg(unix)]
use super::dynamic_root::{DynamicRootFs, DynamicShares};
use super::inode_alloc::InodeAllocator;
use super::null_fs::NullFs;
use super::passthrough::{self, PassthroughFs};
use super::policy::{Deferrals, DeferredRequest};
#[cfg(unix)]
use super::policy::{FsPolicy, PolicyFs};
use super::read_only::PassthroughFsRo;
use super::server::{Server, reply_error};
use super::virtual_entry::VirtualDirEntry;
use crate::virtio::{InterruptTransport, VirtioShmRegion, linux_errno};

/// The most requests a device holds for its policy at a time; each one keeps a descriptor chain
/// of the guest's queue, which is much deeper.
const MAX_DEFERRED: usize = 64;

enum FsServer {
    ReadWrite(Server<AugmentFs<PassthroughFs>>),
    ReadOnly(Server<AugmentFs<PassthroughFsRo>>),
    Null(Server<AugmentFs<NullFs>>),
    #[cfg(unix)]
    Dynamic(Server<AugmentFs<DynamicRootFs>>),
    #[cfg(unix)]
    CheckedDynamic(Server<AugmentFs<PolicyFs<DynamicRootFs>>>),
    #[cfg(unix)]
    CheckedReadWrite(Server<AugmentFs<PolicyFs<PassthroughFs>>>),
    #[cfg(unix)]
    CheckedReadOnly(Server<AugmentFs<PolicyFs<PassthroughFsRo>>>),
}

impl FsServer {
    fn handle_message(
        &self,
        r: Reader,
        w: Writer,
        allow_idmap: bool,
        shm_region: &Option<VirtioShmRegion>,
        exit_code: &Arc<AtomicI32>,
        #[cfg(target_os = "macos")] map_sender: &Option<Sender<WorkerMessage>>,
    ) -> super::Result<usize> {
        macro_rules! dispatch {
            ($($(#[$attr:meta])* $variant:ident),* $(,)?) => {
                match self {
                    $($(#[$attr])* FsServer::$variant(s) => s.handle_message(
                        r,
                        w,
                        allow_idmap,
                        shm_region,
                        exit_code,
                        #[cfg(target_os = "macos")]
                        map_sender,
                    ),)*
                }
            };
        }
        dispatch!(
            ReadWrite,
            ReadOnly,
            Null,
            #[cfg(unix)]
            Dynamic,
            #[cfg(unix)]
            CheckedDynamic,
            #[cfg(unix)]
            CheckedReadWrite,
            #[cfg(unix)]
            CheckedReadOnly,
        )
    }
}

/// A request the policy deferred, whose descriptor chain the device keeps until it is answered.
struct Held {
    queue: usize,
    head: u16,
    request: DeferredRequest,
}

pub struct FsWorker {
    queues: Vec<Queue>,
    queue_evts: Vec<Arc<EventFd>>,
    interrupt: InterruptTransport,
    mem: GuestMemoryMmap,
    allow_idmap: bool,
    shm_region: Option<VirtioShmRegion>,
    server: FsServer,
    stop_fd: EventFd,
    exit_code: Arc<AtomicI32>,
    #[cfg(target_os = "macos")]
    map_sender: Option<Sender<WorkerMessage>>,
    deferrals: Option<Arc<Deferrals>>,
    held: Vec<Held>,
}

impl FsWorker {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        queues: Vec<Queue>,
        queue_evts: Vec<Arc<EventFd>>,
        interrupt: InterruptTransport,
        mem: GuestMemoryMmap,
        allow_idmap: bool,
        shm_region: Option<VirtioShmRegion>,
        passthrough_cfg: Option<passthrough::Config>,
        read_only: bool,
        #[cfg(unix)] dynamic_shares: Option<DynamicShares>,
        #[cfg(unix)] policy: Option<Arc<dyn FsPolicy>>,
        virtual_entries: Vec<VirtualDirEntry<'static>>,
        stop_fd: EventFd,
        exit_code: Arc<AtomicI32>,
        #[cfg(target_os = "macos")] map_sender: Option<Sender<WorkerMessage>>,
    ) -> Result<Self, io::Error> {
        let inode_alloc = Arc::new(InodeAllocator::new());
        #[cfg(unix)]
        let deferrals = match policy {
            Some(_) => Some(Arc::new(Deferrals::new()?)),
            None => None,
        };
        #[cfg(unix)]
        let policy = policy.zip(deferrals.clone());
        // No other host has dynamic shares or checks a policy; this lets one match serve every
        // host.
        #[cfg(not(unix))]
        let (dynamic_shares, policy, deferrals) = (None::<Infallible>, None::<Infallible>, None);
        let server = match (passthrough_cfg, dynamic_shares, policy) {
            #[cfg(unix)]
            (Some(cfg), Some(shares), None) => FsServer::Dynamic(Server::new(AugmentFs::new(
                DynamicRootFs::new(cfg, &shares),
                &inode_alloc,
                virtual_entries,
            ))),
            #[cfg(unix)]
            (Some(cfg), Some(shares), Some((policy, deferrals))) => {
                FsServer::CheckedDynamic(Server::new(AugmentFs::new(
                    PolicyFs::new(DynamicRootFs::new(cfg, &shares), policy, deferrals),
                    &inode_alloc,
                    virtual_entries,
                )))
            }
            #[cfg(unix)]
            (Some(cfg), None, Some((policy, deferrals))) if read_only => {
                let inner = PassthroughFsRo::new(cfg, inode_alloc.clone())?;
                FsServer::CheckedReadOnly(Server::new(AugmentFs::new(
                    PolicyFs::new(inner, policy, deferrals),
                    &inode_alloc,
                    virtual_entries,
                )))
            }
            #[cfg(unix)]
            (Some(cfg), None, Some((policy, deferrals))) => {
                let inner = PassthroughFs::new(cfg, inode_alloc.clone())?;
                FsServer::CheckedReadWrite(Server::new(AugmentFs::new(
                    PolicyFs::new(inner, policy, deferrals),
                    &inode_alloc,
                    virtual_entries,
                )))
            }
            (Some(cfg), None, None) if read_only => {
                let inner = PassthroughFsRo::new(cfg, inode_alloc.clone())?;
                FsServer::ReadOnly(Server::new(AugmentFs::new(
                    inner,
                    &inode_alloc,
                    virtual_entries,
                )))
            }
            (Some(cfg), None, None) => {
                let inner = PassthroughFs::new(cfg, inode_alloc.clone())?;
                FsServer::ReadWrite(Server::new(AugmentFs::new(
                    inner,
                    &inode_alloc,
                    virtual_entries,
                )))
            }
            (None, ..) => FsServer::Null(Server::new(AugmentFs::new(
                NullFs,
                &inode_alloc,
                virtual_entries,
            ))),
        };
        Ok(Self {
            queues,
            queue_evts,
            interrupt,
            mem,
            allow_idmap,
            shm_region,
            server,
            stop_fd,
            exit_code,
            #[cfg(target_os = "macos")]
            map_sender,
            deferrals,
            held: Vec::new(),
        })
    }

    pub fn run(self) -> thread::JoinHandle<()> {
        thread::Builder::new()
            .name("fs worker".into())
            .spawn(|| self.work())
            .unwrap()
    }

    fn work(mut self) {
        let virtq_hpq_ev_fd = self.queue_evts[HPQ_INDEX].as_raw_fd();
        let virtq_req_ev_fd = self.queue_evts[REQ_INDEX].as_raw_fd();
        let stop_ev_fd = self.stop_fd.as_raw_fd();
        let decided_ev_fd = self.deferrals.as_ref().map(|d| d.event.as_raw_fd());

        let mut epoll = Epoll::new().unwrap();

        let _ = epoll.ctl(
            ControlOperation::Add,
            virtq_hpq_ev_fd,
            &EpollEvent::new(EventSet::IN, virtq_hpq_ev_fd as u64),
        );
        let _ = epoll.ctl(
            ControlOperation::Add,
            virtq_req_ev_fd,
            &EpollEvent::new(EventSet::IN, virtq_req_ev_fd as u64),
        );
        let _ = epoll.ctl(
            ControlOperation::Add,
            stop_ev_fd,
            &EpollEvent::new(EventSet::IN, stop_ev_fd as u64),
        );

        if let Some(fd) = decided_ev_fd {
            let _ = epoll.ctl(
                ControlOperation::Add,
                fd,
                &EpollEvent::new(EventSet::IN, fd as u64),
            );
        }

        let mut epoll_events = vec![EpollEvent::new(EventSet::empty(), 0); 32];
        loop {
            match epoll.wait(epoll_events.len(), -1, epoll_events.as_mut_slice()) {
                Ok(ev_cnt) => {
                    for event in &epoll_events[0..ev_cnt] {
                        let source = event.fd();
                        let event_set = event.event_set();
                        match event_set {
                            EventSet::IN if source == virtq_hpq_ev_fd => {
                                self.handle_event(HPQ_INDEX);
                            }
                            EventSet::IN if source == virtq_req_ev_fd => {
                                self.handle_event(REQ_INDEX);
                            }
                            EventSet::IN if Some(source) == decided_ev_fd => {
                                self.serve_decided();
                            }
                            EventSet::IN if source == stop_ev_fd => {
                                debug!("stopping worker thread");
                                let _ = self.stop_fd.read();
                                return;
                            }
                            _ => {
                                log::warn!(
                                    "Received unknown event: {event_set:?} from fd: {source:?}"
                                );
                            }
                        }
                    }
                }
                Err(e) => {
                    debug!("failed to consume muxer epoll event: {e}");
                }
            }
        }
    }

    fn handle_event(&mut self, queue_index: usize) {
        debug!("Fs: queue event: {queue_index}");
        if let Err(e) = self.queue_evts[queue_index].read() {
            error!("Failed to get queue event: {e:?}");
        }

        loop {
            self.queues[queue_index]
                .disable_notification(&self.mem)
                .unwrap();

            self.process_queue(queue_index);

            if !self.queues[queue_index]
                .enable_notification(&self.mem)
                .unwrap()
            {
                break;
            }
        }
    }

    fn process_queue(&mut self, queue_index: usize) {
        while let Some(head) = self.queues[queue_index].pop(&self.mem) {
            let index = head.index;
            let result = self.handle(head);
            self.complete(queue_index, index, result);
        }
    }

    /// Serve again the requests the policy decided.
    fn serve_decided(&mut self) {
        let Some(deferrals) = self.deferrals.clone() else {
            return;
        };
        let _ = deferrals.event.read();
        for id in deferrals.take_decided() {
            let Some(i) = self.held.iter().position(|held| held.request.id() == id) else {
                continue;
            };
            let held = self.held.swap_remove(i);
            let queue = &self.queues[held.queue];
            // The guest may have rewritten the chain meanwhile; it is read again like a new one.
            let Some(head) = DescriptorChain::checked_new(
                &self.mem,
                queue.desc_table,
                queue.actual_size(),
                held.head,
            ) else {
                self.complete(held.queue, held.head, Ok(0));
                continue;
            };
            deferrals.check_again(Some(held.request));
            let result = self.handle(head);
            deferrals.check_again(None);
            self.complete(held.queue, held.head, result);
        }
    }

    fn handle(&self, head: DescriptorChain) -> super::Result<usize> {
        let reader = Reader::new(&self.mem, head.clone())
            .map_err(FsError::QueueReader)
            .unwrap();
        let writer = Writer::new(&self.mem, head)
            .map_err(FsError::QueueWriter)
            .unwrap();
        self.server.handle_message(
            reader,
            writer,
            self.allow_idmap,
            &self.shm_region,
            &self.exit_code,
            #[cfg(target_os = "macos")]
            &self.map_sender,
        )
    }

    fn complete(&mut self, queue_index: usize, head: u16, result: super::Result<usize>) {
        let len = match result {
            Ok(len) => len,
            // The device doesn't offer VIRTIO_F_IN_ORDER, so a request it holds may be completed
            // after the ones that came after it.
            Err(FsError::Deferred { request, .. }) if self.held.len() < MAX_DEFERRED => {
                self.held.push(Held {
                    queue: queue_index,
                    head,
                    request,
                });
                return;
            }
            Err(FsError::Deferred { unique, .. }) => self.refuse(queue_index, head, unique),
            Err(e) => {
                error!("error handling message: {e:?}");
                0
            }
        };

        let queue = &mut self.queues[queue_index];
        if let Err(e) = queue.add_used(&self.mem, head, len as u32) {
            error!("failed to add used elements to the queue: {e:?}");
        }

        if queue.needs_notification(&self.mem).unwrap() {
            self.interrupt.signal_used_queue();
        }
    }

    /// Fail a request the policy deferred when the device holds as many as it can.
    fn refuse(&self, queue_index: usize, head: u16, unique: u64) -> usize {
        let queue = &self.queues[queue_index];
        DescriptorChain::checked_new(&self.mem, queue.desc_table, queue.actual_size(), head)
            .and_then(|head| Writer::new(&self.mem, head).ok())
            .and_then(|writer| reply_error(linux_errno::eacces(), unique, writer).ok())
            .unwrap_or(0)
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::collections::HashMap;
    use std::env;
    use std::fs;
    use std::mem::size_of;
    use std::path::PathBuf;
    use std::process;
    use std::sync::Mutex;
    use std::thread;
    use std::time::{Duration, Instant};

    use utils::eventfd::EFD_NONBLOCK;
    use vm_memory::{Address, ByteValued, Bytes, GuestAddress};

    use super::*;
    use crate::legacy::DummyIrqChip;
    use crate::virtio::fs::fuse::{InHeader, InitInCompat, Opcode, OpenIn, OutHeader, ROOT_ID};
    use crate::virtio::fs::policy::{FsDeferral, FsRequest, FsRequestId, FsVerdict};
    use crate::virtio::fs::{Fs, PermissionSemantics};
    use crate::virtio::queue::tests::VirtQueue;
    use crate::virtio::queue::{VIRTQ_DESC_F_NEXT, VIRTQ_DESC_F_WRITE};
    use crate::virtio::{DeviceQueue, VirtioDevice};

    /// Distinct from anything the passthrough returns, so a refusal is told apart.
    const REFUSED: i32 = 125;
    const QUEUE_SIZE: u16 = 256;
    /// Where the guest puts its requests, each in a slot of its own with room for the reply.
    const SLOTS: u64 = 0x10000;
    const SLOT_SIZE: u64 = 0x200;
    const REPLY_OFFSET: u64 = 0x100;

    /// Defers the requests of one guest process, and lets the others through. A request it
    /// deferred gets the answer the test gives for it.
    struct Defer {
        pid: i32,
        asked: Mutex<Vec<(i32, FsRequestId)>>,
        deferred: Mutex<HashMap<FsRequestId, FsDeferral>>,
        answers: Mutex<HashMap<FsRequestId, FsVerdict>>,
    }

    impl Defer {
        fn new(pid: i32) -> Arc<Self> {
            Arc::new(Self {
                pid,
                asked: Mutex::new(Vec::new()),
                deferred: Mutex::new(HashMap::new()),
                answers: Mutex::new(HashMap::new()),
            })
        }

        /// The ids of the requests it deferred, in the order it was asked about them.
        fn deferred_ids(&self) -> Vec<FsRequestId> {
            let deferred = self.deferred.lock().unwrap();
            let asked = self.asked.lock().unwrap();
            let mut ids: Vec<_> = asked
                .iter()
                .map(|&(_, id)| id)
                .filter(|id| deferred.contains_key(id))
                .collect();
            ids.dedup();
            ids
        }

        fn answer(&self, id: FsRequestId, verdict: FsVerdict) {
            self.answers.lock().unwrap().insert(id, verdict);
            self.deferred.lock().unwrap()[&id].decided();
        }
    }

    impl FsPolicy for Defer {
        fn check(&self, request: &FsRequest) -> FsVerdict {
            self.asked
                .lock()
                .unwrap()
                .push((request.pid(), request.id()));
            if let Some(verdict) = self.answers.lock().unwrap().remove(&request.id()) {
                return verdict;
            }
            if request.pid() != self.pid {
                return FsVerdict::Allow;
            }
            self.deferred
                .lock()
                .unwrap()
                .insert(request.id(), request.deferral());
            FsVerdict::Defer
        }
    }

    struct SharedDir(PathBuf);

    impl SharedDir {
        fn new(tag: &str) -> Self {
            let dir = env::temp_dir().join(format!("libkrun-fs-worker-{tag}-{}", process::id()));
            fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn config(&self) -> passthrough::Config {
            passthrough::Config {
                root_dir: self.0.to_str().unwrap().to_owned(),
                ..Default::default()
            }
        }
    }

    impl Drop for SharedDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// The request queue of a device, filled the way a guest does.
    struct Guest<'a> {
        mem: &'a GuestMemoryMmap,
        queue: VirtQueue<'a>,
        sent: u16,
    }

    impl<'a> Guest<'a> {
        fn new(mem: &'a GuestMemoryMmap) -> Self {
            Self {
                mem,
                queue: VirtQueue::new(GuestAddress(0), mem, QUEUE_SIZE),
                sent: 0,
            }
        }

        fn slot(head: u16) -> GuestAddress {
            GuestAddress(SLOTS + u64::from(head / 2) * SLOT_SIZE)
        }

        /// Put a request in the queue, as a chain of a descriptor for the request and one for
        /// the reply, and return the chain's head.
        fn send<T: ByteValued>(&mut self, opcode: Opcode, pid: u32, args: T) -> u16 {
            let head = self.sent * 2;
            self.sent += 1;
            let len = size_of::<InHeader>() + size_of::<T>();
            let header = InHeader {
                len: len as u32,
                opcode: opcode as u32,
                unique: u64::from(head),
                nodeid: ROOT_ID,
                pid,
                ..Default::default()
            };
            let slot = Self::slot(head);
            self.mem.write_obj(header, slot).unwrap();
            let args_at = slot.unchecked_add(size_of::<InHeader>() as u64);
            self.mem.write_obj(args, args_at).unwrap();
            let reply = slot.unchecked_add(REPLY_OFFSET);
            let dtable = &self.queue.dtable;
            dtable[usize::from(head)].set(slot.0, len as u32, VIRTQ_DESC_F_NEXT, head + 1);
            dtable[usize::from(head) + 1].set(
                reply.0,
                (SLOT_SIZE - REPLY_OFFSET) as u32,
                VIRTQ_DESC_F_WRITE,
                0,
            );
            let idx = self.queue.avail.idx.get();
            self.queue.avail.ring[usize::from(idx % QUEUE_SIZE)].set(head);
            self.queue.avail.idx.set(idx.wrapping_add(1));
            head
        }

        fn init(&mut self) -> u16 {
            let init = InitInCompat {
                major: 7,
                minor: 31,
                ..Default::default()
            };
            self.send(Opcode::Init, 0, init)
        }

        fn opendir(&mut self, pid: u32) -> u16 {
            self.send(Opcode::Opendir, pid, OpenIn::default())
        }

        /// Change the flags of a request the device has, as a guest may while it waits.
        fn rewrite_flags(&self, head: u16, flags: u32) {
            let args_at = Self::slot(head).unchecked_add(size_of::<InHeader>() as u64);
            self.mem
                .write_obj(
                    OpenIn {
                        flags,
                        ..Default::default()
                    },
                    args_at,
                )
                .unwrap();
        }

        /// The heads the device completed, in order, with the errno of each reply.
        fn replies(&self) -> Vec<(u16, i32)> {
            let used = &self.queue.used;
            (0..used.idx.get())
                .map(|i| {
                    let head = used.ring[usize::from(i % QUEUE_SIZE)].get().id as u16;
                    let reply: OutHeader = self
                        .mem
                        .read_obj(Self::slot(head).unchecked_add(REPLY_OFFSET))
                        .unwrap();
                    (head, -reply.error)
                })
                .collect()
        }
    }

    fn memory() -> GuestMemoryMmap {
        GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 0x40000)]).unwrap()
    }

    fn worker(guest: &Guest, dir: &SharedDir, policy: Arc<dyn FsPolicy>) -> FsWorker {
        let queues = vec![Queue::new(QUEUE_SIZE), guest.queue.create_queue()];
        let queue_evts = queues
            .iter()
            .map(|_| Arc::new(EventFd::new(EFD_NONBLOCK).unwrap()))
            .collect();
        FsWorker::new(
            queues,
            queue_evts,
            InterruptTransport::new(DummyIrqChip::new().into(), "fs test".into()).unwrap(),
            guest.mem.clone(),
            false,
            None,
            Some(dir.config()),
            false,
            None,
            Some(policy),
            Vec::new(),
            EventFd::new(EFD_NONBLOCK).unwrap(),
            Arc::new(AtomicI32::new(0)),
            #[cfg(target_os = "macos")]
            None,
        )
        .unwrap()
    }

    #[test]
    fn a_deferred_request_completes_once_decided_and_others_are_served_meanwhile() {
        let mem = memory();
        let mut guest = Guest::new(&mem);
        let dir = SharedDir::new("decided");
        let policy = Defer::new(7);
        let mut worker = worker(&guest, &dir, policy.clone());

        let init = guest.init();
        let held = guest.opendir(7);
        let other = guest.opendir(1);
        worker.process_queue(REQ_INDEX);
        assert_eq!(guest.replies(), [(init, 0), (other, 0)]);

        let later = guest.opendir(1);
        worker.process_queue(REQ_INDEX);
        assert_eq!(guest.replies(), [(init, 0), (other, 0), (later, 0)]);

        let [id] = policy.deferred_ids()[..] else {
            panic!("one request deferred");
        };
        policy.answer(id, FsVerdict::Allow);
        worker.serve_decided();
        assert_eq!(
            guest.replies(),
            [(init, 0), (other, 0), (later, 0), (held, 0)]
        );
        let asked_for_it: Vec<_> = policy
            .asked
            .lock()
            .unwrap()
            .iter()
            .filter(|&&(pid, _)| pid == 7)
            .map(|&(_, id)| id)
            .collect();
        assert_eq!(asked_for_it, [id, id]);
    }

    #[test]
    fn a_request_denied_after_its_deferral_fails_with_the_errno() {
        let mem = memory();
        let mut guest = Guest::new(&mem);
        let dir = SharedDir::new("denied");
        let policy = Defer::new(7);
        let mut worker = worker(&guest, &dir, policy.clone());

        guest.init();
        let held = guest.opendir(7);
        worker.process_queue(REQ_INDEX);
        policy.answer(policy.deferred_ids()[0], FsVerdict::Deny(REFUSED));
        worker.serve_decided();
        assert_eq!(guest.replies().last(), Some(&(held, REFUSED)));
    }

    #[test]
    fn requests_past_the_bound_are_refused_at_once() {
        let mem = memory();
        let mut guest = Guest::new(&mem);
        let dir = SharedDir::new("bound");
        let policy = Defer::new(7);
        let mut worker = worker(&guest, &dir, policy.clone());

        guest.init();
        let heads: Vec<_> = (0..=MAX_DEFERRED).map(|_| guest.opendir(7)).collect();
        worker.process_queue(REQ_INDEX);
        assert_eq!(worker.held.len(), MAX_DEFERRED);
        assert_eq!(guest.replies()[1..], [(heads[MAX_DEFERRED], libc::EACCES)]);

        for id in policy.deferred_ids() {
            policy.answer(id, FsVerdict::Allow);
        }
        worker.serve_decided();
        assert!(worker.held.is_empty());
        let mut replies = guest.replies();
        replies.sort();
        assert_eq!(replies.len(), MAX_DEFERRED + 2);
        assert!(
            replies
                .iter()
                .all(|&(head, errno)| errno == 0 || head == heads[MAX_DEFERRED])
        );
    }

    #[test]
    fn an_answer_holds_only_for_the_request_it_was_given_for() {
        let mem = memory();
        let mut guest = Guest::new(&mem);
        let dir = SharedDir::new("rewritten");
        let policy = Defer::new(7);
        let mut worker = worker(&guest, &dir, policy.clone());

        guest.init();
        let held = guest.opendir(7);
        worker.process_queue(REQ_INDEX);
        let first = policy.deferred_ids()[0];
        guest.rewrite_flags(held, libc::O_NONBLOCK as u32);
        policy.answer(first, FsVerdict::Allow);
        worker.serve_decided();

        assert_eq!(
            guest.replies().len(),
            1,
            "the answer completed another request"
        );
        let [_, second] = policy.deferred_ids()[..] else {
            panic!("the changed request wasn't asked about as a new one");
        };
        assert_ne!(second, first);
        policy.answer(second, FsVerdict::Deny(REFUSED));
        worker.serve_decided();
        assert_eq!(guest.replies().last(), Some(&(held, REFUSED)));
    }

    #[test]
    fn a_reset_drops_the_requests_held() {
        let mem = memory();
        let mut guest = Guest::new(&mem);
        let dir = SharedDir::new("reset");
        let policy = Defer::new(7);
        let mut device = Fs::new(
            "test".into(),
            PermissionSemantics::LinuxComplete,
            Some(dir.config().root_dir),
            Arc::new(AtomicI32::new(0)),
            false,
            Vec::new(),
        )
        .unwrap();
        device.set_policy(policy.clone()).unwrap();
        let notify = Arc::new(EventFd::new(EFD_NONBLOCK).unwrap());
        let queues = vec![
            DeviceQueue::new(
                Queue::new(QUEUE_SIZE),
                Arc::new(EventFd::new(EFD_NONBLOCK).unwrap()),
            ),
            DeviceQueue::new(guest.queue.create_queue(), notify.clone()),
        ];
        let interrupt =
            InterruptTransport::new(DummyIrqChip::new().into(), "fs test".into()).unwrap();
        device.activate(mem.clone(), interrupt, queues).unwrap();

        let wait_for = |what: &str, done: &dyn Fn() -> bool| {
            let deadline = Instant::now() + Duration::from_secs(10);
            while !done() {
                assert!(Instant::now() < deadline, "{what}");
                thread::sleep(Duration::from_millis(1));
            }
        };
        let init = guest.init();
        let answered = guest.opendir(7);
        notify.write(1).unwrap();
        wait_for("the request never reached the policy", &|| {
            policy.deferred_ids().len() == 1
        });
        policy.answer(policy.deferred_ids()[0], FsVerdict::Allow);
        wait_for("the decided request never completed", &|| {
            guest.replies().len() == 2
        });

        guest.opendir(7);
        notify.write(1).unwrap();
        wait_for("the second request never reached the policy", &|| {
            policy.deferred_ids().len() == 2
        });
        assert!(device.reset());
        policy.answer(policy.deferred_ids()[1], FsVerdict::Allow);
        assert_eq!(guest.replies(), [(init, 0), (answered, 0)]);
    }
}
