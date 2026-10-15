/* SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Disk readers for DeepSeek-V4.1's Engram tables on one rank.
 *
 * One worker thread serves every reader's steps in the order the model submits
 * them: per step, the first Engram layer's table, then the second's, so the rows
 * the forward needs first are read first and the two tables never compete for the
 * disk. For each step the worker waits until the GPU has written the step's
 * sequence number into the reader's control block (the stream writes it after
 * copying the row IDs to the host), reads the rows into the mapped buffers the GPU
 * gathers from, and publishes the step by writing its sequence number into the
 * control block. The worker never takes the GIL and never calls CUDA, so nothing
 * another thread does in Python or in the CUDA driver can hold back the rows. A
 * failed step is recorded and not published, and the control block says so: the
 * GPU's wait then stops it at once with a named record instead of letting it read
 * stale rows.
 *
 * Request planning (_row_plan.h) and io_uring execution come from b12x's PLE
 * reader (local-inference-lab/b12x b12x/loader/_ple_reader.c, Apache-2.0): rows
 * are sorted, deduplicated and coalesced into 4 KiB-aligned O_DIRECT reads into
 * registered staging buffers, then scattered to their lookup slots. A short read
 * continues into the rest of its buffer; only the end of the file fails it.
 */
#define _GNU_SOURCE
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <errno.h>
#include <liburing.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    char message[512];
} failure_t;

static void system_error(failure_t *failure, const char *operation) {
    snprintf(failure->message, sizeof(failure->message), "%s: %s", operation, strerror(errno));
}

#include "_row_plan.h"

#define READER_CAPSULE "vllm.engram_rows.reader"
#define ABI_VERSION 2

/* Control block: 32-bit words in mapped host memory, one 64-byte line per writer. */
#define CTRL_IDS_SEQ 0      /* the GPU stream: the step whose row IDs are on the host */
#define CTRL_ROWS_SEQ 16    /* the worker: the step whose rows are in the buffers */
#define CTRL_TIMEOUT_US 32  /* the host: how long the GPU waits for a step's rows */
#define CTRL_WAITED_US 33   /* the GPU: how long a timed-out wait waited */
#define CTRL_FAILED_SEQ 34  /* the GPU: the step a timed-out wait was for */
#define CTRL_PHASE 48       /* the worker: PHASE_* */
#define CTRL_PHASE_SEQ 49   /* the worker: the step it is on */
#define CTRL_WORDS 64

enum { PHASE_IDLE = 0, PHASE_WAITING_IDS = 1, PHASE_READING = 2, PHASE_FAILED = 3 };

#define QUEUE 64u /* steps queued across all readers */

typedef struct {
    char *buffer;
    size_t job;
    unsigned done; /* bytes of the job read so far: a short read continues */
} slot_t;

typedef struct {
    uint32_t sequence;
    uint32_t count;
} step_t;

typedef struct {
    ple_plan_t plan;
    bool poisoned;
    unsigned slots_count;
    slot_t *slots;
    char *buffers;
    struct io_uring ring;
    bool ring_ready;
    bool files_registered;
    int *registered_fds;
    struct iovec *iovecs;
    unsigned *free_slots;
    /* Mapped host buffers, owned by the Python wrapper for the reader's lifetime. */
    const char *ids;
    char *weights, *scales;
    uint32_t *control;
    /* Serving: started (sources fixed) and closing are set once; pending (steps
     * queued or being read) is guarded by the queue's mutex, the rest by mutex. */
    bool started, closing;
    unsigned pending;
    pthread_mutex_t mutex;
    uint64_t ids_timeout_ns, delay_ns;
    failure_t failure; /* the first failure; no step is read after it */
    uint64_t steps_done, lookups_done;
    double ids_wait_seconds, read_seconds;
} reader_t;

static uint64_t now_ns(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static bool reached(uint32_t value, uint32_t target) { return (int32_t)(value - target) >= 0; }

static void control_store(reader_t *reader, int word, uint32_t value) {
    __atomic_store_n(&reader->control[word], value, __ATOMIC_RELAXED);
}

/* --- io_uring execution, from b12x's PLE reader ------------------------------ */

static void ple_error(reader_t *reader, const char *operation, int error) {
    if (!reader->plan.failure.message[0])
        snprintf(reader->plan.failure.message, sizeof(reader->plan.failure.message), "%s: %s",
                 operation, strerror(error));
}

/* Copy a completed job's fragments from its slot buffer to their lookup slots. */
static void ple_scatter(reader_t *reader, size_t index, const char *buffer) {
    ple_job_t *job = &reader->plan.jobs[index];
    for (size_t i = job->begin; i < job->end; i++) {
        ple_fragment_t *fragment = &reader->plan.fragments[i];
        memcpy((fragment->scale ? reader->scales : reader->weights) + fragment->destination,
               buffer + (fragment->offset - job->offset), fragment->length);
    }
}

/* Queue a read of the rest of the slot's job into the rest of its buffer. */
static bool ple_prepare(reader_t *reader, unsigned slot) {
    struct io_uring_sqe *sqe = io_uring_get_sqe(&reader->ring);
    if (!sqe) {
        ple_error(reader, "io_uring SQ capacity", ENOSPC);
        reader->poisoned = true;
        return false;
    }
    slot_t *s = &reader->slots[slot];
    ple_job_t *job = &reader->plan.jobs[s->job];
    io_uring_prep_read_fixed(sqe, (int)job->file, s->buffer + s->done, job->length - s->done,
                             job->offset + s->done, (int)slot);
    sqe->flags |= IOSQE_FIXED_FILE;
    io_uring_sqe_set_data64(sqe, slot);
    return true;
}

/* Submit every prepared SQE. A short positive submit accounts for exactly that many
 * SQEs; retry only the rest. A failed submit may leave prepared SQEs, so stop
 * issuing, drain every submitted CQE, and retire the ring without submitting them. */
static void ple_flush(reader_t *reader, unsigned prepared, unsigned *outstanding) {
    while (prepared) {
        reader->plan.submit_calls++;
        int result = io_uring_submit(&reader->ring);
        if (result == -EINTR) continue;
        if (result <= 0) {
            ple_error(reader, "io_uring submit", result < 0 ? -result : EIO);
            reader->poisoned = true;
            return;
        }
        *outstanding += (unsigned)result;
        prepared -= (unsigned)result;
    }
}

/* Account one completion of the slot's job. Returns whether the slot is free again;
 * a partial read (or EAGAIN/EINTR) queues the rest and *continued counts it. */
static bool ple_complete(reader_t *reader, unsigned slot, int status, unsigned *continued) {
    slot_t *s = &reader->slots[slot];
    ple_job_t *job = &reader->plan.jobs[s->job];
    reader->plan.read_calls++;
    if (status == -EAGAIN || status == -EINTR) {
        if (ple_prepare(reader, slot)) {
            (*continued)++;
            return false;
        }
        return true;
    }
    if (status < 0) {
        ple_error(reader, "io_uring READ_FIXED", -status);
        return true;
    }
    reader->plan.read_bytes += (unsigned)status;
    s->done += (unsigned)status;
    if (s->done >= job->expected) {
        ple_scatter(reader, s->job, s->buffer);
        return true;
    }
    if (status == 0) {
        if (!reader->plan.failure.message[0])
            snprintf(reader->plan.failure.message, sizeof(reader->plan.failure.message),
                     "short Engram row read at offset %lld: the file ended %u bytes early; the "
                     "checkpoint changed or was truncated",
                     (long long)job->offset, job->expected - s->done);
        return true;
    }
    if (ple_prepare(reader, slot)) {
        (*continued)++;
        return false;
    }
    return true;
}

/* Assign the next jobs to free slots and queue their reads; returns how many. */
static unsigned ple_prepare_wave(reader_t *reader, unsigned free_count, size_t *next) {
    size_t remaining = reader->plan.job_count - *next;
    unsigned count = remaining < free_count ? (unsigned)remaining : free_count;
    unsigned prepared = 0;
    for (unsigned i = 0; i < count; i++) {
        unsigned slot = reader->free_slots[i];
        reader->slots[slot].job = (*next)++;
        reader->slots[slot].done = 0;
        if (!ple_prepare(reader, slot)) break;
        prepared++;
    }
    return prepared;
}

static void ple_uring_run(reader_t *reader) {
    unsigned outstanding = 0;
    size_t next = 0;
    for (unsigned i = 0; i < reader->slots_count; i++) reader->free_slots[i] = i;
    ple_flush(reader, ple_prepare_wave(reader, reader->slots_count, &next), &outstanding);
    while (outstanding) {
        struct io_uring_cqe *cqe = NULL;
        /* wait/peek do not submit pending SQEs when draining a failed submit. */
        int result = io_uring_wait_cqe(&reader->ring, &cqe);
        if (result < 0) {
            if (result != -EINTR && result != -EAGAIN) {
                ple_error(reader, "io_uring completion wait", -result);
                reader->poisoned = true;
            }
            continue;
        }
        unsigned seen = 0, free_count = 0, continued = 0, head;
        io_uring_for_each_cqe(&reader->ring, head, cqe) {
            unsigned slot = (unsigned)io_uring_cqe_get_data64(cqe);
            int status = cqe->res;
            seen++;
            outstanding--;
            if (ple_complete(reader, slot, status, &continued)) reader->free_slots[free_count++] = slot;
            if (!outstanding) break;
        }
        io_uring_cq_advance(&reader->ring, seen);
        /* Do not prepare refill SQEs until every collected completion has been
         * checked. An I/O error therefore leaves no speculative refill entries. */
        unsigned prepared = continued;
        if (!reader->plan.failure.message[0] && !reader->poisoned && next < reader->plan.job_count)
            prepared += ple_prepare_wave(reader, free_count, &next);
        ple_flush(reader, prepared, &outstanding);
    }
    if (reader->poisoned) {
        io_uring_queue_exit(&reader->ring);
        reader->ring_ready = false;
    }
}

static int ple_register_files(ple_plan_t *plan, void *owner) {
    (void)plan;
    reader_t *reader = owner;
    int *fds = realloc(reader->registered_fds, reader->plan.file_count * sizeof(*fds));
    if (!fds) return -ENOMEM;
    reader->registered_fds = fds;
    for (size_t i = 0; i < reader->plan.file_count; i++) fds[i] = reader->plan.files[i].fd;
    if (reader->files_registered) {
        int error = io_uring_unregister_files(&reader->ring);
        if (error < 0) {
            reader->poisoned = true;
            return error;
        }
        reader->files_registered = false;
    }
    int error = io_uring_register_files(&reader->ring, fds, (unsigned)reader->plan.file_count);
    if (!error) reader->files_registered = true;
    else reader->poisoned = true;
    return error;
}

/* Read one step's rows: the IDs are in reader->ids, the rows go to their lookup slots. */
static failure_t read_rows(reader_t *reader, uint32_t count) {
    reader->plan.failure.message[0] = 0;
    reader->plan.lookups = count;
    reader->plan.requested_bytes = reader->plan.read_bytes = reader->plan.read_calls = 0;
    reader->plan.unique_blocks = reader->plan.coalesced_reads = reader->plan.submit_calls = 0;
    if (reader->poisoned)
        snprintf(reader->plan.failure.message, sizeof(reader->plan.failure.message),
                 "Engram reader is unusable after an io_uring submission or completion failure");
    else if (ple_plan(&reader->plan, reader->ids, count) &&
             reader->plan.job_count)
        ple_uring_run(reader);
    return reader->plan.failure;
}

/* --- worker ------------------------------------------------------------------ */

/* Wait for the GPU to write the step's row IDs: spin for 200 us (the IDs of a step the
 * GPU is already on arrive within it), then poll every 10 us. */
static bool wait_for_ids(reader_t *reader, uint32_t sequence, failure_t *failure) {
    uint64_t start = now_ns();
    for (;;) {
        if (reached(__atomic_load_n(&reader->control[CTRL_IDS_SEQ], __ATOMIC_ACQUIRE), sequence))
            return true;
        if (__atomic_load_n(&reader->closing, __ATOMIC_RELAXED)) {
            snprintf(failure->message, sizeof(failure->message),
                     "Engram reader closed before the row IDs of step %u arrived", sequence);
            return false;
        }
        if (now_ns() - start > reader->ids_timeout_ns) {
            snprintf(failure->message, sizeof(failure->message),
                     "the row IDs of Engram step %u did not arrive within %.1f s: the GPU is "
                     "blocked before the step's Engram staging",
                     sequence, (double)reader->ids_timeout_ns * 1e-9);
            return false;
        }
        if (now_ns() - start > 200000) {
            struct timespec pause = {0, 10000};
            nanosleep(&pause, NULL);
        } else {
#if defined(__aarch64__)
            __asm__ volatile("yield");
#endif
        }
    }
}

static void serve(reader_t *reader, step_t step) {
    failure_t failure = {{0}};
    control_store(reader, CTRL_PHASE_SEQ, step.sequence);
    control_store(reader, CTRL_PHASE, PHASE_WAITING_IDS);
    uint64_t start = now_ns(), ids_at = start, read_at = start;
    if (wait_for_ids(reader, step.sequence, &failure)) {
        ids_at = now_ns();
        if (reader->delay_ns) {
            struct timespec pause = {(time_t)(reader->delay_ns / 1000000000ull),
                                     (long)(reader->delay_ns % 1000000000ull)};
            nanosleep(&pause, NULL);
        }
        control_store(reader, CTRL_PHASE, PHASE_READING);
        failure = read_rows(reader, step.count);
        read_at = now_ns();
    }
    pthread_mutex_lock(&reader->mutex);
    if (failure.message[0]) {
        if (!reader->failure.message[0]) reader->failure = failure;
    } else {
        reader->steps_done++;
        reader->lookups_done += step.count;
        reader->ids_wait_seconds += (double)(ids_at - start) * 1e-9;
        reader->read_seconds += (double)(read_at - ids_at) * 1e-9;
    }
    pthread_mutex_unlock(&reader->mutex);
    if (failure.message[0]) {
        control_store(reader, CTRL_PHASE, PHASE_FAILED);
        return;
    }
    /* The rows are in place before the GPU can see the step. */
    __atomic_store_n(&reader->control[CTRL_ROWS_SEQ], step.sequence, __ATOMIC_RELEASE);
    control_store(reader, CTRL_PHASE, PHASE_IDLE);
}

/* The process's one worker and its queue of (reader, step) jobs. */
typedef struct {
    reader_t *reader;
    step_t step;
} job_t;

static struct {
    pthread_mutex_t mutex;
    pthread_cond_t wake, space, idle;
    job_t jobs[QUEUE];
    uint64_t head, tail; /* tail - head jobs are queued */
    bool started;
} queue = {.mutex = PTHREAD_MUTEX_INITIALIZER};

static pthread_once_t queue_once = PTHREAD_ONCE_INIT;

static void queue_init(void) {
    pthread_condattr_t monotonic;
    pthread_condattr_init(&monotonic);
    pthread_condattr_setclock(&monotonic, CLOCK_MONOTONIC);
    pthread_cond_init(&queue.wake, &monotonic);
    pthread_cond_init(&queue.space, &monotonic);
    pthread_cond_init(&queue.idle, &monotonic);
    pthread_condattr_destroy(&monotonic);
}

static void *worker_main(void *arg) {
    (void)arg;
    /* Sleep as asked: the default 50 us timer slack would stretch every 10 us poll. */
    prctl(PR_SET_TIMERSLACK, 1UL, 0, 0, 0);
    pthread_mutex_lock(&queue.mutex);
    for (;;) {
        while (queue.head == queue.tail) pthread_cond_wait(&queue.wake, &queue.mutex);
        job_t job = queue.jobs[queue.head % QUEUE];
        queue.head++;
        pthread_cond_broadcast(&queue.space);
        pthread_mutex_unlock(&queue.mutex);
        pthread_mutex_lock(&job.reader->mutex);
        bool failed = job.reader->failure.message[0] != 0;
        pthread_mutex_unlock(&job.reader->mutex);
        if (!failed) serve(job.reader, job.step);
        pthread_mutex_lock(&queue.mutex);
        job.reader->pending--;
        pthread_cond_broadcast(&queue.idle);
    }
    return NULL;
}

/* Close the reader: its queued steps fail at once, then it is freed. */
static void release(reader_t *reader) {
    if (!reader) return;
    pthread_mutex_lock(&queue.mutex);
    __atomic_store_n(&reader->closing, true, __ATOMIC_RELAXED);
    while (reader->pending) pthread_cond_wait(&queue.idle, &queue.mutex);
    pthread_mutex_unlock(&queue.mutex);
    if (reader->ring_ready) io_uring_queue_exit(&reader->ring);
    free(reader->iovecs);
    free(reader->free_slots);
    free(reader->registered_fds);
    ple_plan_release(&reader->plan);
    free(reader->buffers);
    free(reader->slots);
    pthread_mutex_destroy(&reader->mutex);
    free(reader);
}

static void delete_capsule(PyObject *capsule) {
    reader_t *reader = PyCapsule_GetPointer(capsule, READER_CAPSULE);
    if (reader) {
        Py_BEGIN_ALLOW_THREADS
        release(reader);
        Py_END_ALLOW_THREADS
    }
}

/* --- Python surface ---------------------------------------------------------- */

static reader_t *get_reader(PyObject *capsule) {
    return PyCapsule_GetPointer(capsule, READER_CAPSULE);
}

/* reader(table_rows, shard_start, shard_end, max_lookups, queue_depth, ids, weights,
 *        scales, control, ids_timeout_s, delay_s): the four buffers are addresses of
 * host memory that outlives the reader. */
static PyObject *py_reader(PyObject *self, PyObject *args) {
    (void)self;
    long long table_rows, shard_start, shard_end, max_lookups;
    int depth;
    unsigned long long ids, weights, scales, control;
    double ids_timeout, delay;
    if (!PyArg_ParseTuple(args, "LLLLiKKKKdd", &table_rows, &shard_start, &shard_end, &max_lookups,
                          &depth, &ids, &weights, &scales, &control, &ids_timeout, &delay))
        return NULL;
    if (table_rows <= 0 || shard_start < 0 || shard_end < shard_start || shard_end > table_rows ||
        max_lookups <= 0 || max_lookups > UINT32_MAX || depth <= 0 || depth > 32768 || !ids ||
        !weights || !scales || !control || !(ids_timeout > 0) || delay < 0)
        return PyErr_Format(PyExc_ValueError, "invalid Engram reader geometry, buffers or times");
    reader_t *reader = calloc(1, sizeof(*reader));
    if (!reader) return PyErr_NoMemory();
    pthread_mutex_init(&reader->mutex, NULL);
    /* One whole-table source per plane: shard 0 spans every row. */
    int error = ple_plan_init(&reader->plan, table_rows, table_rows, shard_start, shard_end, 256, 8,
                              max_lookups);
    if (error) {
        release(reader);
        return error == ENOMEM ? PyErr_NoMemory()
                               : PyErr_Format(PyExc_ValueError, "Engram row plan: %s", strerror(error));
    }
    reader->ids = (const char *)(uintptr_t)ids;
    reader->weights = (char *)(uintptr_t)weights;
    reader->scales = (char *)(uintptr_t)scales;
    reader->control = (uint32_t *)(uintptr_t)control;
    reader->ids_timeout_ns = (uint64_t)(ids_timeout * 1e9);
    reader->delay_ns = (uint64_t)(delay * 1e9);
    reader->slots_count = (unsigned)depth;
    reader->slots = calloc(reader->slots_count, sizeof(*reader->slots));
    error = posix_memalign((void **)&reader->buffers, PLE_BLOCK, (size_t)reader->slots_count * PLE_READ_MAX);
    if (!reader->slots || error) {
        release(reader);
        return PyErr_NoMemory();
    }
    for (unsigned i = 0; i < reader->slots_count; i++)
        reader->slots[i].buffer = reader->buffers + (size_t)i * PLE_READ_MAX;
    error = io_uring_queue_init(reader->slots_count, &reader->ring, 0);
    if (error < 0) {
        release(reader);
        return PyErr_Format(PyExc_RuntimeError,
                            "io_uring initialization failed: %s; enable io_uring in the kernel and "
                            "container policy",
                            strerror(-error));
    }
    reader->ring_ready = true;
    reader->iovecs = calloc(reader->slots_count, sizeof(*reader->iovecs));
    reader->free_slots = calloc(reader->slots_count, sizeof(*reader->free_slots));
    if (!reader->iovecs || !reader->free_slots) {
        release(reader);
        return PyErr_NoMemory();
    }
    for (unsigned i = 0; i < reader->slots_count; i++) {
        reader->iovecs[i].iov_base = reader->slots[i].buffer;
        reader->iovecs[i].iov_len = PLE_READ_MAX;
    }
    error = io_uring_register_buffers(&reader->ring, reader->iovecs, reader->slots_count);
    if (error < 0) {
        release(reader);
        return PyErr_Format(PyExc_RuntimeError,
                            "io_uring buffer registration failed: %s; raise the locked-memory limit "
                            "or reduce queue_depth",
                            strerror(-error));
    }
    PyObject *capsule = PyCapsule_New(reader, READER_CAPSULE, delete_capsule);
    if (!capsule) release(reader);
    return capsule;
}

/* add(reader, path, offset, scale): register the table's weight or scale plane. */
static PyObject *py_add(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule;
    const char *path;
    long long offset;
    int scale;
    if (!PyArg_ParseTuple(args, "OsLp", &capsule, &path, &offset, &scale)) return NULL;
    reader_t *reader = get_reader(capsule);
    if (!reader) return NULL;
    int error;
    failure_t failure = {{0}};
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&reader->mutex);
    if (__atomic_load_n(&reader->started, __ATOMIC_RELAXED)) {
        error = EBUSY;
    } else {
        reader->plan.failure.message[0] = 0;
        error = ple_add_source(&reader->plan, 0, path, offset, scale != 0, ple_register_files, reader);
        failure = reader->plan.failure;
    }
    pthread_mutex_unlock(&reader->mutex);
    Py_END_ALLOW_THREADS
    if (error == EBUSY) return PyErr_Format(PyExc_RuntimeError, "Engram sources are fixed once steps run");
    if (failure.message[0]) return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    if (error) return PyErr_Format(PyExc_ValueError, "Engram source: %s", strerror(error));
    Py_RETURN_NONE;
}

/* submit(reader, sequence, count, wait_s): queue one step behind every step already
 * queued. Returns at once unless the queue is full; then it waits for space, without
 * the GIL. */
static PyObject *py_submit(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule;
    unsigned int sequence, count;
    double wait;
    if (!PyArg_ParseTuple(args, "OIId", &capsule, &sequence, &count, &wait)) return NULL;
    reader_t *reader = get_reader(capsule);
    if (!reader) return NULL;
    if (count > reader->plan.max_lookups)
        return PyErr_Format(PyExc_ValueError, "Engram step of %u lookups exceeds the reader's capacity", count);
    const char *problem = NULL;
    failure_t failure = {{0}};
    Py_BEGIN_ALLOW_THREADS
    struct timespec deadline;
    clock_gettime(CLOCK_MONOTONIC, &deadline);
    deadline.tv_sec += (time_t)wait;
    deadline.tv_nsec += (long)((wait - (double)(time_t)wait) * 1e9);
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec++;
        deadline.tv_nsec -= 1000000000L;
    }
    pthread_mutex_lock(&queue.mutex);
    for (;;) {
        pthread_mutex_lock(&reader->mutex);
        failure = reader->failure;
        pthread_mutex_unlock(&reader->mutex);
        if (failure.message[0] || queue.tail - queue.head < QUEUE) break;
        if (pthread_cond_timedwait(&queue.space, &queue.mutex, &deadline) == ETIMEDOUT) break;
    }
    if (!failure.message[0]) {
        if (queue.tail - queue.head >= QUEUE) {
            problem = "Engram readers fell more than 64 steps behind";
        } else if (!queue.started) {
            pthread_t thread;
            pthread_attr_t attr;
            pthread_attr_init(&attr);
            pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
            if (pthread_create(&thread, &attr, worker_main, NULL)) problem = "Engram reader thread did not start";
            else queue.started = true;
            pthread_attr_destroy(&attr);
        }
        if (!problem) {
            __atomic_store_n(&reader->started, true, __ATOMIC_RELAXED);
            queue.jobs[queue.tail % QUEUE] = (job_t){reader, {sequence, count}};
            queue.tail++;
            reader->pending++;
            pthread_cond_signal(&queue.wake);
        }
    }
    pthread_mutex_unlock(&queue.mutex);
    Py_END_ALLOW_THREADS
    if (failure.message[0]) return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    if (problem) return PyErr_Format(PyExc_RuntimeError, "%s", problem);
    Py_RETURN_NONE;
}

/* status(reader): the first failure (or None) and the worker's counters. */
static PyObject *py_status(PyObject *self, PyObject *capsule) {
    (void)self;
    reader_t *reader = get_reader(capsule);
    if (!reader) return NULL;
    failure_t failure;
    uint64_t steps, lookups, queued;
    double ids_wait, read;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&queue.mutex);
    queued = reader->pending;
    pthread_mutex_unlock(&queue.mutex);
    pthread_mutex_lock(&reader->mutex);
    failure = reader->failure;
    steps = reader->steps_done;
    lookups = reader->lookups_done;
    ids_wait = reader->ids_wait_seconds;
    read = reader->read_seconds;
    pthread_mutex_unlock(&reader->mutex);
    Py_END_ALLOW_THREADS
    PyObject *message = failure.message[0] ? PyUnicode_FromString(failure.message) : Py_NewRef(Py_None);
    if (!message) return NULL;
    PyObject *result = Py_BuildValue("{s:N,s:K,s:K,s:K,s:d,s:d}", "failure", message, "steps",
                                     (unsigned long long)steps, "lookups", (unsigned long long)lookups,
                                     "queued", (unsigned long long)queued, "ids_wait_seconds", ids_wait,
                                     "read_seconds", read);
    return result;
}

static PyMethodDef methods[] = {
    {"reader", py_reader, METH_VARARGS, NULL},
    {"add", py_add, METH_VARARGS, NULL},
    {"submit", py_submit, METH_VARARGS, NULL},
    {"status", py_status, METH_O, NULL},
    {NULL, NULL, 0, NULL},
};

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT, .m_name = "_engram_rows_reader", .m_size = -1, .m_methods = methods,
};

PyMODINIT_FUNC PyInit__engram_rows_reader(void) {
    pthread_once(&queue_once, queue_init);
    PyObject *result = PyModule_Create(&module);
    if (!result) return NULL;
    if (PyModule_AddIntConstant(result, "ABI_VERSION", ABI_VERSION) < 0 ||
        PyModule_AddIntConstant(result, "CTRL_IDS_SEQ", CTRL_IDS_SEQ) < 0 ||
        PyModule_AddIntConstant(result, "CTRL_ROWS_SEQ", CTRL_ROWS_SEQ) < 0 ||
        PyModule_AddIntConstant(result, "CTRL_TIMEOUT_US", CTRL_TIMEOUT_US) < 0 ||
        PyModule_AddIntConstant(result, "CTRL_WAITED_US", CTRL_WAITED_US) < 0 ||
        PyModule_AddIntConstant(result, "CTRL_FAILED_SEQ", CTRL_FAILED_SEQ) < 0 ||
        PyModule_AddIntConstant(result, "CTRL_PHASE", CTRL_PHASE) < 0 ||
        PyModule_AddIntConstant(result, "CTRL_PHASE_SEQ", CTRL_PHASE_SEQ) < 0 ||
        PyModule_AddIntConstant(result, "CTRL_WORDS", CTRL_WORDS) < 0) {
        Py_DECREF(result);
        return NULL;
    }
    return result;
}
