// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Storage-owned buffer exports for the pipeline package.
//!
//! The owner is the actual CPython buffer export, not a public memoryview,
//! NumPy array, or Tensor wrapper. Aliases must retain that export until their
//! native work finishes. This does not track CUDA completion or other processes.
//!
//! # Safety invariants
//!
//! CPython calls these functions with an attached thread state. Factory-created
//! objects have the exact, immutable, non-subclassable heap types in their
//! interpreter's module state. Initialization finishes before publication;
//! afterwards only the ticket's atomic phase changes. Deallocation exclusively
//! owns the object, and releases storage before invoking Python bookkeeping.
//! Buffer addresses are never turned into Rust references or slices: Python and
//! native consumers can alias and mutate them. `_export_owned` trusts its caller
//! to supply a live writable address/extent and the corresponding allocator.
//!
//! Exporters are deliberately untracked and GC clear is a no-op. Their callback
//! and allocator graphs must survive cyclic GC/resurrection until destruction.
//! This conservatively retains ownership cycles rather than invalidating aliases.
//! All fallible work uses the C API's error protocol; these C entrypoints contain
//! no panicking Rust operations, Rust heap allocations, or user Rust closures.

// Build-time API provider choice; both paths use only the same raw CPython ABI.
use std::cell::UnsafeCell;
use std::ffi::CStr;
use std::ffi::c_char;
use std::ffi::c_int;
use std::ffi::c_void;
use std::mem::size_of;
use std::mem::transmute;
use std::ptr::addr_of_mut;
use std::ptr::null;
use std::ptr::null_mut;
use std::sync::Once;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering;

#[cfg(pyo3_ffi_reexport)]
use pyo3::ffi;
#[cfg(not(pyo3_ffi_reexport))]
use pyo3_ffi as ffi;

const INACTIVE: u8 = 0;
const SETUP: u8 = 1;
const NATIVE: u8 = 2;
const RELEASED: u8 = 3;
const STATE_EMPTY: u8 = 0;
const STATE_INITIALIZING: u8 = 1;
const STATE_READY: u8 = 2;

#[repr(C)]
struct Ticket {
    head: ffi::PyObject,
    phase: AtomicU8,
    release: *mut ffi::PyObject,
}

#[repr(C)]
struct Exporter {
    head: ffi::PyObject,
    backing: ffi::Py_buffer,
    release: *mut ffi::PyObject,
    owner: *mut ffi::PyObject,
    ticket: bool,
}

#[repr(C)]
struct State {
    exporter_type: *mut ffi::PyTypeObject,
    ticket_type: *mut ffi::PyTypeObject,
    // CPython zero-initializes aligned module state; zero is a valid AtomicU8.
    // Acquire/Release publishes both type pointers to reentrant/FT factories.
    status: AtomicU8,
}

// Method definitions outlive their Python callables. CPython only reads these
// arrays, including when interpreters execute concurrently.
struct Methods<const N: usize>([ffi::PyMethodDef; N]);
// SAFETY: Every pointer is to immutable static text/code; the array is read-only.
unsafe impl<const N: usize> Sync for Methods<N> {}

const fn method(
    name: &'static CStr,
    function: ffi::PyCFunction,
    flags: c_int,
    doc: &'static CStr,
) -> ffi::PyMethodDef {
    ffi::PyMethodDef {
        ml_name: name.as_ptr(),
        ml_meth: ffi::PyMethodDefPointer {
            PyCFunction: function,
        },
        ml_flags: flags,
        ml_doc: doc.as_ptr(),
    }
}

static TICKET_METHODS: Methods<4> = Methods([
    method(
        c"_activate",
        ticket_activate,
        ffi::METH_NOARGS,
        c"Acquire setup ownership.",
    ),
    method(
        c"_cancel",
        ticket_cancel,
        ffi::METH_NOARGS,
        c"Cancel without notifying; reject native ownership.",
    ),
    method(
        c"_rollback",
        ticket_rollback,
        ffi::METH_NOARGS,
        c"Retire setup ownership once; leave native ownership intact.",
    ),
    ffi::PyMethodDef::zeroed(),
]);

static METHODS: Methods<5> = Methods([
    method(
        c"_release_ticket",
        make_ticket,
        ffi::METH_O,
        c"Create an inactive private release ticket.",
    ),
    method(
        c"_export",
        export_buffer,
        ffi::METH_VARARGS,
        c"Export writable contiguous bytes with storage-owned release.",
    ),
    method(
        c"_export_owned",
        export_owned,
        ffi::METH_VARARGS,
        c"Export a trusted allocator address while retaining its owner.",
    ),
    method(
        c"_is_exporter",
        is_exporter,
        ffi::METH_O,
        c"Recognize this module instance's native exporter.",
    ),
    ffi::PyMethodDef::zeroed(),
]);

// SAFETY: object points to a live CPython object or is null; callers own the ref.
unsafe fn clear_reference(object: *mut *mut ffi::PyObject) {
    let previous = unsafe { object.replace(null_mut()) };
    unsafe { ffi::Py_XDECREF(previous) };
}

// SAFETY: CPython has exclusively entered tp_dealloc. The immutable type was
// checked at module creation to have the alloc/free slots used here. Generic
// heap-type allocation owns one type reference, which must outlive tp_free.
unsafe fn free_heap_object(object: *mut ffi::PyObject) {
    unsafe {
        let ty = ffi::Py_TYPE(object);
        let free: ffi::freefunc = transmute(ffi::PyType_GetSlot(ty, ffi::Py_tp_free));
        free(object.cast());
        ffi::Py_DECREF(ty.cast());
    }
}

// SAFETY: Exact factory-created Ticket; phase is initialized and release is
// immutable while any caller/native exporter retains the ticket.
unsafe fn release_ticket(ticket: *mut Ticket, from: u8) -> *mut ffi::PyObject {
    unsafe {
        if (*ticket)
            .phase
            .compare_exchange(from, RELEASED, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return ffi::Py_NewRef(ffi::Py_None());
        }
        ffi::PyObject_CallNoArgs((*ticket).release)
    }
}

unsafe extern "C" fn ticket_activate(
    object: *mut ffi::PyObject,
    _: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    // SAFETY: Immutable method descriptor enforces the exact Ticket receiver.
    unsafe {
        if (*object.cast::<Ticket>())
            .phase
            .compare_exchange(INACTIVE, SETUP, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            ffi::PyErr_SetString(
                ffi::PyExc_RuntimeError,
                c"export ticket is already activated".as_ptr(),
            );
            return null_mut();
        }
        ffi::Py_NewRef(ffi::Py_None())
    }
}

unsafe extern "C" fn ticket_cancel(
    object: *mut ffi::PyObject,
    _: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    // SAFETY: Ticket's only concurrently mutable member is its atomic phase.
    unsafe {
        let phase = &(*object.cast::<Ticket>()).phase;
        let mut current = phase.load(Ordering::SeqCst);
        while current != RELEASED {
            if current == NATIVE {
                ffi::PyErr_SetString(
                    ffi::PyExc_RuntimeError,
                    c"native storage owns the export ticket".as_ptr(),
                );
                return null_mut();
            }
            match phase.compare_exchange_weak(current, RELEASED, Ordering::SeqCst, Ordering::SeqCst)
            {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
        ffi::Py_NewRef(ffi::Py_None())
    }
}

unsafe extern "C" fn ticket_rollback(
    object: *mut ffi::PyObject,
    _: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    // SAFETY: The Ticket method descriptor validates the receiver.
    unsafe { release_ticket(object.cast(), SETUP) }
}

unsafe extern "C" fn ticket_dealloc(object: *mut ffi::PyObject) {
    // SAFETY: Exclusive destruction; no callback is run merely by dropping a
    // ticket. Clearing a reference before DECREF also makes reentry harmless.
    unsafe {
        ffi::PyObject_GC_UnTrack(object.cast());
        let ticket = object.cast::<Ticket>();
        clear_reference(addr_of_mut!((*ticket).release));
        std::ptr::drop_in_place(addr_of_mut!((*ticket).phase));
        free_heap_object(object);
    }
}

// SAFETY: Referents remain live during CPython traversal; return the visitor's
// first nonzero result just as Py_VISIT does.
unsafe fn visit_references<const N: usize>(
    objects: [*mut ffi::PyObject; N],
    visit: ffi::visitproc,
    arg: *mut c_void,
) -> c_int {
    for object in objects {
        if !object.is_null() {
            let result = unsafe { visit(object, arg) };
            if result != 0 {
                return result;
            }
        }
    }
    0
}

unsafe extern "C" fn ticket_traverse(
    object: *mut ffi::PyObject,
    visit: ffi::visitproc,
    arg: *mut c_void,
) -> c_int {
    // SAFETY: CPython coordinates GC traversal; these referents are immutable.
    unsafe {
        visit_references(
            [
                ffi::Py_TYPE(object).cast(),
                (*object.cast::<Ticket>()).release,
            ],
            visit,
            arg,
        )
    }
}

unsafe extern "C" fn exporter_traverse(
    object: *mut ffi::PyObject,
    visit: ffi::visitproc,
    arg: *mut c_void,
) -> c_int {
    // SAFETY: The exporter is initialized before publication and remains intact
    // until exclusive deallocation, even if a view is resurrected by cyclic GC.
    unsafe {
        let exporter = object.cast::<Exporter>();
        visit_references(
            [
                ffi::Py_TYPE(object).cast(),
                (*exporter).backing.obj,
                (*exporter).release,
                (*exporter).owner,
            ],
            visit,
            arg,
        )
    }
}

unsafe extern "C" fn preserve_ownership(_: *mut ffi::PyObject) -> c_int {
    // GC is not evidence that a retained/resurrected alias has finished.
    0
}

// These public stable-ABI symbols predate PyErr_GetRaisedException (3.12).
// Bind them explicitly to keep error preservation usable on 3.10/3.11 and lower
// abi3 targets without version-dependent deprecation lint suppressions. Fetch
// transfers three owned references out; Restore steals those same references.
unsafe extern "C" {
    #[link_name = "PyErr_Fetch"]
    fn fetch_error(
        ty: *mut *mut ffi::PyObject,
        value: *mut *mut ffi::PyObject,
        traceback: *mut *mut ffi::PyObject,
    );
    #[link_name = "PyErr_Restore"]
    fn restore_error(
        ty: *mut ffi::PyObject,
        value: *mut ffi::PyObject,
        traceback: *mut ffi::PyObject,
    );
}

unsafe extern "C" fn exporter_dealloc(object: *mut ffi::PyObject) {
    // SAFETY: Exclusive destruction. Preserve an already-pending Python error
    // through backing/owner destruction, the callback, and unraisable reporting.
    unsafe {
        ffi::PyObject_GC_UnTrack(object.cast());
        let mut error_type = null_mut();
        let mut error_value = null_mut();
        let mut error_traceback = null_mut();
        fetch_error(&mut error_type, &mut error_value, &mut error_traceback);
        let exporter = object.cast::<Exporter>();
        ffi::PyBuffer_Release(addr_of_mut!((*exporter).backing));
        clear_reference(addr_of_mut!((*exporter).owner));
        let release = (*exporter).release;
        if !release.is_null() {
            let result = if (*exporter).ticket {
                release_ticket(release.cast(), NATIVE)
            } else {
                ffi::PyObject_CallNoArgs(release)
            };
            if result.is_null() {
                ffi::PyErr_WriteUnraisable(release);
            }
            ffi::Py_XDECREF(result);
            clear_reference(addr_of_mut!((*exporter).release));
        }
        free_heap_object(object);
        restore_error(error_type, error_value, error_traceback);
    }
}

unsafe extern "C" fn get_buffer(
    object: *mut ffi::PyObject,
    view: *mut ffi::Py_buffer,
    flags: c_int,
) -> c_int {
    // SAFETY: FillInfo retains object, so each consumer's Py_buffer keeps the
    // immutable address and ownership graph alive. No Rust data reference exists.
    unsafe {
        let exporter = object.cast::<Exporter>();
        ffi::PyBuffer_FillInfo(
            view,
            object,
            (*exporter).backing.buf,
            (*exporter).backing.len,
            0,
            flags,
        )
    }
}

unsafe extern "C" fn reject_new(
    _: *mut ffi::PyTypeObject,
    _: *mut ffi::PyObject,
    _: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    // SAFETY: An attached CPython thread can set the exception indicator.
    unsafe {
        ffi::PyErr_SetString(
            ffi::PyExc_TypeError,
            c"storage exporters can only be created by their factories".as_ptr(),
        )
    };
    null_mut()
}

// SAFETY: CPython supplies a live module; state can be null before exec.
unsafe fn module_state(module: *mut ffi::PyObject) -> *mut State {
    unsafe { ffi::PyModule_GetState(module).cast() }
}

// SAFETY: The factory's bound module is live. Its function table may be callable
// before exec (including GC reentry during type creation), so no type/alloc slot
// is used until both module-owned types have been published successfully.
unsafe fn checked_state(module: *mut ffi::PyObject) -> *mut State {
    unsafe {
        let state = module_state(module);
        if state.is_null()
            || (*state).status.load(Ordering::Acquire) != STATE_READY
            || (*state).exporter_type.is_null()
            || (*state).ticket_type.is_null()
        {
            ffi::PyErr_SetString(
                ffi::PyExc_RuntimeError,
                c"native module is not initialized".as_ptr(),
            );
            return null_mut();
        }
        state
    }
}

// SAFETY: Type is an immutable module-owned heap type, checked during exec.
unsafe fn allocate(ty: *mut ffi::PyTypeObject) -> *mut ffi::PyObject {
    unsafe {
        let alloc: ffi::allocfunc = transmute(ffi::PyType_GetSlot(ty, ffi::Py_tp_alloc));
        alloc(ty, 0)
    }
}

// SAFETY: State passed the ready guard; no initialized exporter escapes until
// after all its fields and callback ownership are installed.
unsafe fn allocate_exporter(state: *mut State) -> *mut Exporter {
    unsafe {
        let object = allocate((*state).exporter_type);
        if object.is_null() {
            return null_mut();
        }
        ffi::PyObject_GC_UnTrack(object.cast());
        let exporter = object.cast::<Exporter>();
        addr_of_mut!((*exporter).backing).write(ffi::Py_buffer::new());
        addr_of_mut!((*exporter).release).write(null_mut());
        addr_of_mut!((*exporter).owner).write(null_mut());
        addr_of_mut!((*exporter).ticket).write(false);
        exporter
    }
}

unsafe extern "C" fn make_ticket(
    module: *mut ffi::PyObject,
    release: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    // SAFETY: METH_O supplies one live borrowed argument. No allocation or
    // Python callback occurs between alloc and atomic/reference initialization.
    unsafe {
        let state = checked_state(module);
        if state.is_null() {
            return null_mut();
        }
        if ffi::PyCallable_Check(release) == 0 {
            ffi::PyErr_SetString(ffi::PyExc_TypeError, c"release must be callable".as_ptr());
            return null_mut();
        }
        let object = allocate((*state).ticket_type);
        if !object.is_null() {
            let ticket = object.cast::<Ticket>();
            addr_of_mut!((*ticket).phase).write(AtomicU8::new(INACTIVE));
            addr_of_mut!((*ticket).release).write(ffi::Py_NewRef(release));
        }
        object
    }
}

unsafe extern "C" fn export_buffer(
    module: *mut ffi::PyObject,
    args: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    // SAFETY: METH_VARARGS supplies a tuple; O stores borrowed PyObject pointers.
    // The backing Py_buffer is uniquely held here and later by the exporter.
    unsafe {
        let state = checked_state(module);
        if state.is_null() {
            return null_mut();
        }
        let mut source: *mut ffi::PyObject = null_mut();
        let mut release: *mut ffi::PyObject = null_mut();
        if ffi::PyArg_ParseTuple(args, c"OO:_export".as_ptr(), &mut source, &mut release) == 0 {
            return null_mut();
        }
        let ticket = ffi::Py_TYPE(release) == (*state).ticket_type;
        if !ticket && ffi::PyCallable_Check(release) == 0 {
            ffi::PyErr_SetString(
                ffi::PyExc_TypeError,
                c"release must be callable or an export ticket".as_ptr(),
            );
            return null_mut();
        }
        let exporter = allocate_exporter(state);
        if exporter.is_null() {
            return null_mut();
        }
        if ffi::PyObject_GetBuffer(
            source,
            addr_of_mut!((*exporter).backing),
            ffi::PyBUF_WRITABLE | ffi::PyBUF_C_CONTIGUOUS,
        ) < 0
        {
            ffi::Py_DECREF(exporter.cast());
            return null_mut();
        }
        let mut view = ffi::PyMemoryView_FromObject(exporter.cast());
        if !view.is_null() {
            (*exporter).release = ffi::Py_NewRef(release);
            (*exporter).ticket = ticket;
            if ticket
                && (*release.cast::<Ticket>())
                    .phase
                    .compare_exchange(SETUP, NATIVE, Ordering::SeqCst, Ordering::SeqCst)
                    .is_err()
            {
                ffi::PyErr_SetString(
                    ffi::PyExc_RuntimeError,
                    c"export ticket is not setup-owned".as_ptr(),
                );
                // Rejected reuse must never retire another exporter's capability.
                clear_reference(addr_of_mut!((*exporter).release));
                (*exporter).ticket = false;
                clear_reference(&mut view);
            }
        }
        ffi::Py_DECREF(exporter.cast());
        view
    }
}

unsafe extern "C" fn export_owned(
    module: *mut ffi::PyObject,
    args: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    // SAFETY: OnO parses a Python address object, Py_ssize_t, and owner. The
    // caller, not integer conversion, guarantees this writable allocation exists.
    unsafe {
        let state = checked_state(module);
        if state.is_null() {
            return null_mut();
        }
        let mut address_object: *mut ffi::PyObject = null_mut();
        let mut owner: *mut ffi::PyObject = null_mut();
        let mut size: ffi::Py_ssize_t = 0;
        if ffi::PyArg_ParseTuple(
            args,
            c"OnO:_export_owned".as_ptr(),
            &mut address_object,
            &mut size,
            &mut owner,
        ) == 0
        {
            return null_mut();
        }
        let address = ffi::PyLong_AsVoidPtr(address_object);
        if !ffi::PyErr_Occurred().is_null() {
            return null_mut();
        }
        if size < 0 || (size != 0 && (address.is_null() || owner == ffi::Py_None())) {
            ffi::PyErr_SetString(
                ffi::PyExc_ValueError,
                c"nonempty storage requires an address and owner".as_ptr(),
            );
            return null_mut();
        }
        let exporter = allocate_exporter(state);
        if exporter.is_null() {
            return null_mut();
        }
        (*exporter).backing.buf = address;
        (*exporter).backing.len = size;
        (*exporter).owner = ffi::Py_NewRef(owner);
        let view = ffi::PyMemoryView_FromObject(exporter.cast());
        ffi::Py_DECREF(exporter.cast());
        view
    }
}

unsafe extern "C" fn is_exporter(
    module: *mut ffi::PyObject,
    object: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    // SAFETY: Exact type identity is interpreter/module-local, not name-based.
    unsafe {
        let state = checked_state(module);
        if state.is_null() {
            return null_mut();
        }
        ffi::PyBool_FromLong((ffi::Py_TYPE(object) == (*state).exporter_type).into())
    }
}

// SAFETY: Stack spec/slot arrays are consumed synchronously by FromSpec. Method
// definitions, unlike specs/slots, are static and remain live with descriptors.
unsafe fn make_type(
    name: *const c_char,
    basicsize: usize,
    slots: *mut ffi::PyType_Slot,
) -> *mut ffi::PyTypeObject {
    unsafe {
        let mut spec = ffi::PyType_Spec {
            name,
            basicsize: basicsize as c_int,
            itemsize: 0,
            flags: (ffi::Py_TPFLAGS_DEFAULT
                | ffi::Py_TPFLAGS_HAVE_GC
                | ffi::Py_TPFLAGS_IMMUTABLETYPE) as _,
            slots,
        };
        let ty = ffi::PyType_FromSpec(&mut spec).cast::<ffi::PyTypeObject>();
        if !ty.is_null()
            && (ffi::PyType_GetSlot(ty, ffi::Py_tp_alloc).is_null()
                || ffi::PyType_GetSlot(ty, ffi::Py_tp_free).is_null())
        {
            ffi::Py_DECREF(ty.cast());
            ffi::PyErr_SetString(
                ffi::PyExc_SystemError,
                c"native heap type has no allocation/free slot".as_ptr(),
            );
            return null_mut();
        }
        ty
    }
}

const fn slot(id: c_int, function: *mut c_void) -> ffi::PyType_Slot {
    ffi::PyType_Slot {
        slot: id,
        pfunc: function,
    }
}

unsafe extern "C" fn exec_module(module: *mut ffi::PyObject) -> c_int {
    // SAFETY: CPython allocates aligned, zero-initialized State before exec.
    // AtomicU8 accepts the zero representation. Factories see neither type until
    // successful publication; repeated/reentrant exec cannot overwrite owners.
    unsafe {
        let state = module_state(module);
        if state.is_null() {
            ffi::PyErr_SetString(
                ffi::PyExc_RuntimeError,
                c"native module has no state".as_ptr(),
            );
            return -1;
        }
        match (*state).status.compare_exchange(
            STATE_EMPTY,
            STATE_INITIALIZING,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {}
            Err(STATE_READY) => return 0,
            Err(_) => {
                ffi::PyErr_SetString(
                    ffi::PyExc_RuntimeError,
                    c"native module is initializing".as_ptr(),
                );
                return -1;
            }
        }
        let result = initialize_types(module, state);
        if result < 0 {
            (*state).status.store(STATE_EMPTY, Ordering::Release);
        } else {
            (*state).status.store(STATE_READY, Ordering::Release);
        }
        result
    }
}

// SAFETY: This exec exclusively owns STATE_INITIALIZING. Type references remain
// local until both types and the module attribute are successfully installed.
unsafe fn initialize_types(module: *mut ffi::PyObject, state: *mut State) -> c_int {
    unsafe {
        let mut exporter_slots = [
            slot(ffi::Py_tp_new, reject_new as *mut c_void),
            slot(ffi::Py_tp_dealloc, exporter_dealloc as *mut c_void),
            slot(ffi::Py_tp_traverse, exporter_traverse as *mut c_void),
            slot(ffi::Py_tp_clear, preserve_ownership as *mut c_void),
            slot(ffi::Py_bf_getbuffer, get_buffer as *mut c_void),
            slot(0, null_mut()),
        ];
        let exporter_type = make_type(
            c"spdl.pipeline.lib._native._Exporter".as_ptr(),
            size_of::<Exporter>(),
            exporter_slots.as_mut_ptr(),
        );
        if exporter_type.is_null() {
            return -1;
        }
        let mut ticket_slots = [
            slot(ffi::Py_tp_new, reject_new as *mut c_void),
            slot(ffi::Py_tp_dealloc, ticket_dealloc as *mut c_void),
            slot(ffi::Py_tp_traverse, ticket_traverse as *mut c_void),
            slot(ffi::Py_tp_clear, preserve_ownership as *mut c_void),
            slot(
                ffi::Py_tp_methods,
                TICKET_METHODS.0.as_ptr().cast_mut().cast(),
            ),
            slot(0, null_mut()),
        ];
        let ticket_type = make_type(
            c"spdl.pipeline.lib._native._ReleaseTicket".as_ptr(),
            size_of::<Ticket>(),
            ticket_slots.as_mut_ptr(),
        );
        if ticket_type.is_null() {
            ffi::Py_DECREF(exporter_type.cast());
            return -1;
        }
        if ffi::PyModule_AddObjectRef(module, c"_ReleaseTicket".as_ptr(), ticket_type.cast()) < 0 {
            ffi::Py_DECREF(ticket_type.cast());
            ffi::Py_DECREF(exporter_type.cast());
            return -1;
        }
        (*state).exporter_type = exporter_type;
        (*state).ticket_type = ticket_type;
        0
    }
}

unsafe extern "C" fn traverse_module(
    module: *mut ffi::PyObject,
    visit: ffi::visitproc,
    arg: *mut c_void,
) -> c_int {
    // SAFETY: CPython calls traverse even for partially executed modules.
    unsafe {
        let state = module_state(module);
        if state.is_null() {
            return 0;
        }
        visit_references(
            [(*state).exporter_type.cast(), (*state).ticket_type.cast()],
            visit,
            arg,
        )
    }
}

unsafe extern "C" fn clear_module(module: *mut ffi::PyObject) -> c_int {
    // SAFETY: Module clearing is coordinated by CPython, including exec failure.
    unsafe {
        let state = module_state(module);
        if !state.is_null() {
            (*state).status.store(STATE_EMPTY, Ordering::Release);
            clear_reference(addr_of_mut!((*state).exporter_type).cast());
            clear_reference(addr_of_mut!((*state).ticket_type).cast());
        }
    }
    0
}

unsafe extern "C" fn free_module(module: *mut c_void) {
    // SAFETY: CPython supplies the dying module (not its state); clear is idempotent.
    unsafe {
        clear_module(module.cast());
    }
}

struct ModuleSlots<const N: usize>([ffi::PyModuleDef_Slot; N]);
// SAFETY: Slots contain only static function pointers and CPython's ABI sentinels.
unsafe impl<const N: usize> Sync for ModuleSlots<N> {}

// Stable CPython module slot ABI (Include/moduleobject.h): slot 3, introduced in
// 3.12, value 2 declares per-interpreter GIL support; slot 4, introduced in 3.13,
// value 1 declares no GIL requirement. Runtime selection avoids depending on
// cfgs that pyo3-ffi's build script does not propagate to this crate, and avoids
// presenting newer slots to older CPython/limited-API builds.
static SLOTS_OLD: ModuleSlots<2> = ModuleSlots([
    ffi::PyModuleDef_Slot {
        slot: ffi::Py_mod_exec,
        value: exec_module as *mut c_void,
    },
    ffi::PyModuleDef_Slot {
        slot: 0,
        value: null_mut(),
    },
]);
static SLOTS_312: ModuleSlots<3> = ModuleSlots([
    ffi::PyModuleDef_Slot {
        slot: ffi::Py_mod_exec,
        value: exec_module as *mut c_void,
    },
    ffi::PyModuleDef_Slot {
        slot: 3,
        value: 2usize as *mut c_void,
    },
    ffi::PyModuleDef_Slot {
        slot: 0,
        value: null_mut(),
    },
]);
static SLOTS_313: ModuleSlots<4> = ModuleSlots([
    ffi::PyModuleDef_Slot {
        slot: ffi::Py_mod_exec,
        value: exec_module as *mut c_void,
    },
    ffi::PyModuleDef_Slot {
        slot: 3,
        value: 2usize as *mut c_void,
    },
    ffi::PyModuleDef_Slot {
        slot: 4,
        value: std::ptr::without_provenance_mut(1),
    },
    ffi::PyModuleDef_Slot {
        slot: 0,
        value: null_mut(),
    },
]);

struct ModuleDefinition(UnsafeCell<ffi::PyModuleDef>);
// SAFETY: Once serializes initialization/slot selection. Subsequently only
// CPython accesses the definition under its extension/module-definition locks;
// all our per-interpreter mutable data lives in PyModule_GetState instead.
unsafe impl Sync for ModuleDefinition {}

static MODULE_INIT: Once = Once::new();
static MODULE_READY: AtomicBool = AtomicBool::new(false);
static MODULE_DEF: ModuleDefinition = ModuleDefinition(UnsafeCell::new(ffi::PyModuleDef {
    m_base: ffi::PyModuleDef_HEAD_INIT,
    m_name: c"_native".as_ptr(),
    m_doc: null(),
    m_size: size_of::<State>() as ffi::Py_ssize_t,
    m_methods: METHODS.0.as_ptr().cast_mut(),
    m_slots: null_mut(),
    m_traverse: Some(traverse_module),
    m_clear: Some(clear_module),
    m_free: Some(free_module),
}));

/// CPython multi-phase entrypoint, intentionally not a cached PyO3 `pymodule`.
///
/// # Safety
/// Must be invoked by CPython's extension loader with an attached thread state.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyInit__native() -> *mut ffi::PyObject {
    MODULE_INIT.call_once(|| {
        // SAFETY: Py_GetVersion is process-static NUL-terminated text. No Rust
        // operation here allocates or panics, so Once cannot become poisoned.
        unsafe {
            let version = CStr::from_ptr(ffi::Py_GetVersion()).to_bytes();
            let minor = version
                .split(|byte| *byte == b'.')
                .nth(1)
                .and_then(|part| std::str::from_utf8(part).ok())
                .and_then(|part| part.parse::<u32>().ok());
            let Some(minor) = minor.filter(|minor| version.starts_with(b"3.") && *minor >= 10)
            else {
                ffi::PyErr_SetString(
                    ffi::PyExc_ImportError,
                    c"native buffer exports require CPython 3.10 or newer in the 3.x series"
                        .as_ptr(),
                );
                return;
            };
            (*MODULE_DEF.0.get()).m_slots = if minor >= 13 {
                SLOTS_313.0.as_ptr().cast_mut()
            } else if minor >= 12 {
                SLOTS_312.0.as_ptr().cast_mut()
            } else {
                SLOTS_OLD.0.as_ptr().cast_mut()
            };
            MODULE_READY.store(
                !ffi::PyModuleDef_Init(MODULE_DEF.0.get()).is_null(),
                Ordering::Release,
            );
        }
    });
    if !MODULE_READY.load(Ordering::Acquire) {
        // SAFETY: Preserve the original init error, or report the cached failure
        // to a later importing thread/interpreter rather than returning a bad def.
        unsafe {
            if ffi::PyErr_Occurred().is_null() {
                ffi::PyErr_SetString(
                    ffi::PyExc_ImportError,
                    c"native module definition initialization failed".as_ptr(),
                );
            }
        }
        return null_mut();
    }
    MODULE_DEF.0.get().cast()
}
