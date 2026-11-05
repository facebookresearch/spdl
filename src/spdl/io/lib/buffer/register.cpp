/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Shared arena reuse/unmapping must follow the lifetime of the actual buffer
 * export, not a weakref/finalizer on the original Tensor or memoryview wrapper.
 * Detached tensors, retained storage, NumPy bases and memoryview aliases can
 * outlive that wrapper. Releasing one public view is not proof that a native
 * consumer has finished; consumers must retain their buffer export/owner.
 *
 * Our supported Python 3.11 has no Python-defined buffer-export hooks. Python
 * 3.12+ offers __buffer__/__release_buffer__ (PEP 688), but a portable
 * replacement would need a separately validated storage-owner design, not just
 * a wrapper finalizer. This exporter supplies that owner through the buffer
 * protocol.
 *
 * _export retains a writable contiguous Py_buffer and transfers the release
 * callback only after successful view construction. On exporter destruction,
 * it releases the backing export before calling Python lease bookkeeping.
 * _export_owned instead retains the allocator owner for an internal address and
 * byte extent; it is not a general pointer-validity checker. Conservative GC
 * clearing avoids invalidating resurrected aliases, not a promise to collect
 * every ownership cycle. Neither path tracks CUDA completion or cross-process
 * slot accounting; callers must retain storage owners until pending work ends.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <atomic>
#include <new>

namespace {
enum class Phase { Inactive, Setup, Native, Released };
struct Ticket {
  PyObject_HEAD std::atomic<Phase> phase;
  PyObject* release;
};
struct Exporter {
  PyObject_HEAD Py_buffer backing{};
  PyObject* release{};
  PyObject* owner{};
  bool ticket{};
};
struct State {
  PyTypeObject* type;
  PyTypeObject* ticketType;
};

int ticketTraverse(Ticket* self, visitproc visit, void* arg) {
  Py_VISIT(Py_TYPE(self));
  Py_VISIT(self->release);
  return 0;
}

int ticketClear(Ticket*) {
  // A retained/resurrected native export still owns this callback capability.
  return 0;
}

void ticketDealloc(Ticket* self) {
  PyObject_GC_UnTrack(self);
  Py_CLEAR(self->release);
  self->phase.~atomic();
  PyTypeObject* type = Py_TYPE(self);
  type->tp_free(reinterpret_cast<PyObject*>(self));
  Py_DECREF(type);
}

PyObject* ticketActivate(Ticket* self, PyObject*) {
  Phase expected = Phase::Inactive;
  if (!self->phase.compare_exchange_strong(expected, Phase::Setup)) {
    PyErr_SetString(PyExc_RuntimeError, "export ticket is already activated");
    return nullptr;
  }
  Py_RETURN_NONE;
}

PyObject* ticketCancel(Ticket* self, PyObject*) {
  Phase phase = self->phase.load();
  while (phase != Phase::Released) {
    if (phase == Phase::Native) {
      PyErr_SetString(
          PyExc_RuntimeError, "native storage owns the export ticket");
      return nullptr;
    }
    if (self->phase.compare_exchange_weak(phase, Phase::Released)) {
      break;
    }
  }
  Py_RETURN_NONE;
}

PyObject* releaseTicket(Ticket* self, Phase from) {
  if (!self->phase.compare_exchange_strong(from, Phase::Released)) {
    Py_RETURN_NONE;
  }
  return PyObject_CallNoArgs(self->release);
}

PyObject* ticketRollback(Ticket* self, PyObject*) {
  return releaseTicket(self, Phase::Setup);
}

int traverse(Exporter* self, visitproc visit, void* arg) {
  Py_VISIT(Py_TYPE(self));
  Py_VISIT(self->backing.obj);
  Py_VISIT(self->release);
  Py_VISIT(self->owner);
  return 0;
}

int clear(Exporter*) {
  // Clearing storage before tp_dealloc could invalidate resurrected native
  // aliases.
  return 0;
}

void dealloc(Exporter* self) {
  PyObject_GC_UnTrack(self);
  PyObject *errorType, *errorValue, *errorTraceback;
  PyErr_Fetch(&errorType, &errorValue, &errorTraceback);
  // Drop the backing export/allocator owner before bookkeeping permits reuse.
  PyBuffer_Release(&self->backing);
  Py_CLEAR(self->owner);
  if (self->release) {
    PyObject* result = self->ticket
        ? releaseTicket(reinterpret_cast<Ticket*>(self->release), Phase::Native)
        : PyObject_CallNoArgs(self->release);
    if (!result) {
      PyErr_WriteUnraisable(self->release);
    }
    Py_XDECREF(result);
    Py_CLEAR(self->release);
  }
  PyTypeObject* type = Py_TYPE(self);
  type->tp_free(reinterpret_cast<PyObject*>(self));
  Py_DECREF(type);
  PyErr_Restore(errorType, errorValue, errorTraceback);
}

int getbuffer(Exporter* self, Py_buffer* view, int flags) {
  return PyBuffer_FillInfo(
      view,
      reinterpret_cast<PyObject*>(self),
      self->backing.buf,
      self->backing.len,
      0,
      flags);
}

PyObject* rejectNew(PyTypeObject*, PyObject*, PyObject*) {
  PyErr_SetString(
      PyExc_TypeError,
      "storage exporters can only be created by their factories");
  return nullptr;
}

PyType_Slot typeSlots[] = {
    {Py_tp_new, reinterpret_cast<void*>(rejectNew)},
    {Py_tp_dealloc, reinterpret_cast<void*>(dealloc)},
    {Py_tp_traverse, reinterpret_cast<void*>(traverse)},
    {Py_tp_clear, reinterpret_cast<void*>(clear)},
    {Py_bf_getbuffer, reinterpret_cast<void*>(getbuffer)},
    {0, nullptr}};
PyType_Spec typeSpec = {
    "spdl.io.lib._buffer._Exporter",
    sizeof(Exporter),
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_IMMUTABLETYPE,
    typeSlots};

PyMethodDef ticketMethods[] = {
    {"_activate",
     reinterpret_cast<PyCFunction>(ticketActivate),
     METH_NOARGS,
     nullptr},
    {"_cancel",
     reinterpret_cast<PyCFunction>(ticketCancel),
     METH_NOARGS,
     nullptr},
    {"_rollback",
     reinterpret_cast<PyCFunction>(ticketRollback),
     METH_NOARGS,
     nullptr},
    {nullptr, nullptr, 0, nullptr}};
PyType_Slot ticketSlots[] = {
    {Py_tp_new, reinterpret_cast<void*>(rejectNew)},
    {Py_tp_dealloc, reinterpret_cast<void*>(ticketDealloc)},
    {Py_tp_traverse, reinterpret_cast<void*>(ticketTraverse)},
    {Py_tp_clear, reinterpret_cast<void*>(ticketClear)},
    {Py_tp_methods, reinterpret_cast<void*>(ticketMethods)},
    {0, nullptr}};
PyType_Spec ticketSpec = {
    "spdl.io.lib._buffer._ReleaseTicket",
    sizeof(Ticket),
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_IMMUTABLETYPE,
    ticketSlots};

PyObject* makeTicket(PyObject* module, PyObject* release) {
  if (!PyCallable_Check(release)) {
    PyErr_SetString(PyExc_TypeError, "release must be callable");
    return nullptr;
  }
  auto* state = static_cast<State*>(PyModule_GetState(module));
  auto* self = reinterpret_cast<Ticket*>(
      state->ticketType->tp_alloc(state->ticketType, 0));
  if (!self) {
    return nullptr;
  }
  new (&self->phase) std::atomic<Phase>(Phase::Inactive);
  self->release = Py_NewRef(release);
  return reinterpret_cast<PyObject*>(self);
}

Exporter* allocateExporter(PyObject* module) {
  auto* state = static_cast<State*>(PyModule_GetState(module));
  auto* self =
      reinterpret_cast<Exporter*>(state->type->tp_alloc(state->type, 0));
  if (self) {
    // Keep the callback/allocator graph alive until actual export destruction.
    // Cyclic GC must not clear a Python partial or bound owner before dealloc
    // calls it. This is deliberately conservative for callback ownership
    // cycles; memoryviews and ordinary/resurrected aliases still retain this
    // exporter.
    PyObject_GC_UnTrack(self);
  }
  return self;
}

PyObject* exportBuffer(PyObject* module, PyObject* args) {
  PyObject *source, *release;
  if (!PyArg_ParseTuple(args, "OO:_export", &source, &release)) {
    return nullptr;
  }
  auto* state = static_cast<State*>(PyModule_GetState(module));
  bool ticket = Py_TYPE(release) == state->ticketType;
  if (!ticket && !PyCallable_Check(release)) {
    PyErr_SetString(
        PyExc_TypeError, "release must be callable or an export ticket");
    return nullptr;
  }
  auto* self = allocateExporter(module);
  if (!self) {
    return nullptr;
  }
  if (PyObject_GetBuffer(
          source, &self->backing, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) < 0) {
    Py_DECREF(self);
    return nullptr;
  }
  PyObject* view = PyMemoryView_FromObject(reinterpret_cast<PyObject*>(self));
  if (view) {
    // The native ticket is adopted before any successful view reaches Python.
    // This capability, not return status, decides whether setup may roll back.
    self->release = Py_NewRef(release);
    self->ticket = ticket;
    if (ticket) {
      Phase expected = Phase::Setup;
      if (!reinterpret_cast<Ticket*>(release)->phase.compare_exchange_strong(
              expected, Phase::Native)) {
        PyErr_SetString(PyExc_RuntimeError, "export ticket is not setup-owned");
        // A rejected reuse must not release another exporter's native ticket.
        Py_CLEAR(self->release);
        self->ticket = false;
        Py_CLEAR(view);
      }
    }
  }
  Py_DECREF(self);
  return view;
}

PyObject* exportOwned(PyObject* module, PyObject* args) {
  PyObject *addressObject, *owner;
  Py_ssize_t size;
  if (!PyArg_ParseTuple(
          args, "OnO:_export_owned", &addressObject, &size, &owner)) {
    return nullptr;
  }
  void* address = PyLong_AsVoidPtr(addressObject);
  if (PyErr_Occurred()) {
    return nullptr;
  }
  if (size < 0 || (size && (!address || owner == Py_None))) {
    PyErr_SetString(
        PyExc_ValueError, "nonempty storage requires an address and owner");
    return nullptr;
  }
  auto* self = allocateExporter(module);
  if (!self) {
    return nullptr;
  }
  self->backing.buf = address;
  self->backing.len = size;
  self->owner = Py_NewRef(owner);
  PyObject* view = PyMemoryView_FromObject(reinterpret_cast<PyObject*>(self));
  Py_DECREF(self);
  return view;
}

PyMethodDef methods[] = {
    {"_release_ticket",
     makeTicket,
     METH_O,
     "Create an inactive private export release ticket."},
    {"_export",
     exportBuffer,
     METH_VARARGS,
     "Export bytes with a native storage-owned release callback."},
    {"_export_owned",
     exportOwned,
     METH_VARARGS,
     "Export an internal allocator address while retaining its owning object."},
    {nullptr, nullptr, 0, nullptr}};

int execModule(PyObject* module) {
  auto* state = static_cast<State*>(PyModule_GetState(module));
  state->type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&typeSpec));
  if (!state->type) {
    return -1;
  }
  state->ticketType =
      reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&ticketSpec));
  if (!state->ticketType) {
    return -1;
  }
  return PyModule_AddObjectRef(
      module, "_ReleaseTicket", reinterpret_cast<PyObject*>(state->ticketType));
}
int traverseModule(PyObject* module, visitproc visit, void* arg) {
  auto* state = static_cast<State*>(PyModule_GetState(module));
  Py_VISIT(state->type);
  Py_VISIT(state->ticketType);
  return 0;
}
int clearModule(PyObject* module) {
  auto* state = static_cast<State*>(PyModule_GetState(module));
  Py_CLEAR(state->type);
  Py_CLEAR(state->ticketType);
  return 0;
}
PyModuleDef_Slot moduleSlots[] = {
    {Py_mod_exec, reinterpret_cast<void*>(execModule)},
#if PY_VERSION_HEX >= 0x030C0000
    {Py_mod_multiple_interpreters, Py_MOD_PER_INTERPRETER_GIL_SUPPORTED},
#endif
#if PY_VERSION_HEX >= 0x030D0000
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, nullptr}};
PyModuleDef moduleDef = {
    PyModuleDef_HEAD_INIT,
    "_buffer",
    nullptr,
    sizeof(State),
    methods,
    moduleSlots,
    traverseModule,
    clearModule,
    nullptr};
} // namespace

PyMODINIT_FUNC PyInit__buffer();

PyMODINIT_FUNC PyInit__buffer() {
  return PyModuleDef_Init(&moduleDef);
}
