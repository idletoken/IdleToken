/* Deflated sealed payloads (platform → provider agent).
 *
 * A relay job for an agentic request is mostly tool schemas — tens of KB of
 * repetitive JSON — and it has to cross whatever link the provider's home has.
 * Measured 2026-09-13 on a real one: a 365 KiB job took three minutes and often
 * did not arrive at all, while the same machine pulled 274 KB over HTTPS in two
 * seconds. Deflating the plaintext before sealing it takes that payload to
 * roughly a seventh of its size, which is the difference between "works" and
 * "the provider drops off the market while it reads".
 *
 * The marker lives INSIDE the envelope rather than beside it in the wire JSON:
 * the envelope is already the unit both sides agree on, and an agent that
 * predates this reads exactly the bytes it always read. A JSON body cannot
 * begin with these four bytes, so the discriminator is unambiguous, and the
 * platform only ever sends a deflated payload to an agent that advertised it
 * (`accepts_deflate` in its capacity) — the marker is how the agent recognizes
 * what it asked for, not a negotiation in itself.
 */
#ifndef IDLETOKEN_DEFLATE_WIRE_H
#define IDLETOKEN_DEFLATE_WIRE_H

#include <stddef.h>
#include <stdint.h>

#define IDLETOKEN_DEFLATE_MAGIC     "ITZ1"
#define IDLETOKEN_DEFLATE_MAGIC_LEN 4

/* Does this opened envelope carry a deflated payload? */
int idletoken_wire_is_deflated(const void *buf, size_t len);

/* Inflate a marked payload into a fresh buffer (caller frees with
 * idletoken_wire_free so the plaintext is wiped, never plain free()).
 *
 * `max_out` is a hard ceiling on the inflated size, checked BEFORE anything is
 * allocated: a decompressor without one is a memory amplifier even when the
 * sender is honest, and this one runs on someone's home machine.
 *
 * Returns 0 on success; non-zero leaves *out NULL. */
int idletoken_wire_inflate(const void *in, size_t in_len,
                           uint8_t **out, size_t *out_len, size_t max_out);

/* Wipe and release a buffer handed back by idletoken_wire_inflate. */
void idletoken_wire_free(uint8_t *buf, size_t len);

#endif /* IDLETOKEN_DEFLATE_WIRE_H */
