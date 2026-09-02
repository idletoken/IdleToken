#!/usr/bin/env python3
"""Independent minisign/rsign signature verifier — pure Python, no dependencies.

Why this file exists, given that the client already verifies signatures:

  The client verifies with the same library that produced the signature
  (`tauri-plugin-updater` -> `minisign-verify`, signed by `tauri signer` ->
  `rsign2`). A checker that shares its implementation with the thing it checks
  cannot notice a shared mistake, and it cannot be used by anybody who has not
  installed our software yet — which is exactly the person who most needs to
  check (DIST-03: the first install is the one the update signature does not
  cover).

  So this is a second implementation, written from RFC 8032 and the minisign
  file format, that depends on nothing but CPython. Anyone can read it, run it
  against an artifact they downloaded, and get an answer that does not come
  from us at runtime.

Formats understood
------------------
Public key file (or a bare base64 blob, or the base64-of-the-whole-file form
that `tauri.conf.json` `plugins.updater.pubkey` uses):

    untrusted comment: minisign public key: <KEYID-HEX>
    base64( b"Ed" || key_id[8] || public_key[32] )

Signature file (Tauri stores the whole file base64-encoded again inside
`latest.json`; both forms are accepted):

    untrusted comment: <text>
    base64( alg[2] || key_id[8] || signature[64] )
    trusted comment: <text>
    base64( global_signature[64] )

`alg` is b"Ed" (sign the message as-is) or b"ED" (sign BLAKE2b-512 of the
message). Both are implemented; guessing which one a producer used is not our
business.

The trusted comment is checked too: its signature covers
`signature || trusted_comment`, so an attacker who can rewrite the human
-readable "trusted comment: file:..., timestamp:..." line without invalidating
anything would otherwise be able to relabel a valid signature.

Exit codes:  0 verified · 1 verification failed · 2 usage/format error.

Usage:
    minisign_verify.py --pubkey <file-or-b64> --sig <file> <artifact>
    minisign_verify.py --pubkey <file-or-b64> --sig <file> --json <artifact>
    minisign_verify.py --self-test
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import sys

# ---------------------------------------------------------------------------
# Ed25519 verification (RFC 8032, §5.1.7). Reference-style implementation:
# clarity over speed, because the whole point is that a stranger can read it
# and agree it is Ed25519.
# ---------------------------------------------------------------------------

_P = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493
_D = (-121665 * pow(121666, _P - 2, _P)) % _P
_I = pow(2, (_P - 1) // 4, _P)  # sqrt(-1)


def _recover_x(y: int, sign: int) -> int | None:
    """The x coordinate of a compressed point, or None if the point is not on
    the curve. `sign` is the low bit the encoding carried."""
    if y >= _P:
        return None
    xx = (y * y - 1) * pow(_D * y * y + 1, _P - 2, _P) % _P
    x = pow(xx, (_P + 3) // 8, _P)
    if (x * x - xx) % _P != 0:
        x = (x * _I) % _P
    if (x * x - xx) % _P != 0:
        return None
    if x % 2 != sign:
        x = _P - x
    return x


# Extended homogeneous coordinates (X, Y, Z, T) with x = X/Z, y = Y/Z, xy = T/Z.
_BY = 4 * pow(5, _P - 2, _P) % _P
_BX = _recover_x(_BY, 0)
assert _BX is not None
_B = (_BX, _BY, 1, _BX * _BY % _P)
_IDENT = (0, 1, 1, 0)


def _add(p, q):
    px, py, pz, pt = p
    qx, qy, qz, qt = q
    a = (py - px) * (qy - qx) % _P
    b = (py + px) * (qy + qx) % _P
    c = 2 * pt * qt * _D % _P
    d = 2 * pz * qz % _P
    e, f, g, h = b - a, d - c, d + c, b + a
    return (e * f % _P, g * h % _P, f * g % _P, e * h % _P)


def _mul(p, n: int):
    r = _IDENT
    while n > 0:
        if n & 1:
            r = _add(r, p)
        p = _add(p, p)
        n >>= 1
    return r


def _equal(p, q) -> bool:
    px, py, pz, _ = p
    qx, qy, qz, _ = q
    return (px * qz - qx * pz) % _P == 0 and (py * qz - qy * pz) % _P == 0


def _decompress(b: bytes):
    if len(b) != 32:
        return None
    y = int.from_bytes(b, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    if x is None:
        return None
    return (x, y, 1, x * y % _P)


def ed25519_verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
    """True iff `signature` is a valid Ed25519 signature of `message`."""
    if len(public_key) != 32 or len(signature) != 64:
        return False
    a = _decompress(public_key)
    if a is None:
        return False
    r_bytes, s_bytes = signature[:32], signature[32:]
    r = _decompress(r_bytes)
    if r is None:
        return False
    s = int.from_bytes(s_bytes, "little")
    # Non-canonical S is a malleability hole, not a curiosity: rejecting it is
    # part of the spec.
    if s >= _L:
        return False
    k = int.from_bytes(
        hashlib.sha512(r_bytes + public_key + message).digest(), "little"
    ) % _L
    return _equal(_mul(_B, s), _add(r, _mul(a, k)))


# ---------------------------------------------------------------------------
# minisign container parsing
# ---------------------------------------------------------------------------


class FormatError(Exception):
    pass


def _maybe_unwrap_base64(text: str) -> str:
    """Tauri stores whole key/signature FILES base64-encoded (the `pubkey`
    field of tauri.conf.json and the `signature` field of latest.json are both
    that shape). Unwrap one layer when the blob decodes to something that
    starts like a minisign file; otherwise leave it alone."""
    stripped = "".join(text.split())
    if not stripped:
        return text
    try:
        decoded = base64.b64decode(stripped, validate=True)
    except (binascii.Error, ValueError):
        return text
    try:
        inner = decoded.decode("utf-8")
    except UnicodeDecodeError:
        return text
    if inner.lstrip().startswith("untrusted comment:"):
        return inner
    return text


def _read_blob(value: str) -> str:
    """`value` is either a path to a file or the literal content."""
    if os.path.isfile(value):
        with open(value, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    return value


def parse_public_key(value: str) -> tuple[bytes, bytes]:
    """Return (key_id, public_key_bytes)."""
    text = _maybe_unwrap_base64(_read_blob(value))
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise FormatError("empty public key")
    payload = lines[-1] if lines[0].startswith("untrusted comment:") else lines[0]
    try:
        raw = base64.b64decode("".join(payload.split()), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise FormatError(f"public key is not base64: {exc}") from exc
    if len(raw) != 42:
        raise FormatError(
            f"public key payload is {len(raw)} bytes, expected 42 "
            "(2 alg + 8 key id + 32 key)"
        )
    if raw[:2] != b"Ed":
        raise FormatError(f"unsupported public key algorithm {raw[:2]!r}")
    return raw[2:10], raw[10:]


def parse_signature(value: str) -> dict:
    text = _maybe_unwrap_base64(_read_blob(value))
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    if len(lines) < 2:
        raise FormatError("signature file has fewer than two lines")
    if not lines[0].startswith("untrusted comment:"):
        raise FormatError("signature file does not start with an untrusted comment")
    try:
        sig_raw = base64.b64decode("".join(lines[1].split()), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise FormatError(f"signature line is not base64: {exc}") from exc
    if len(sig_raw) != 74:
        raise FormatError(
            f"signature payload is {len(sig_raw)} bytes, expected 74 "
            "(2 alg + 8 key id + 64 signature)"
        )
    alg = sig_raw[:2]
    if alg not in (b"Ed", b"ED"):
        raise FormatError(f"unsupported signature algorithm {alg!r}")
    out = {
        "alg": alg,
        "key_id": sig_raw[2:10],
        "signature": sig_raw[10:],
        "untrusted_comment": lines[0][len("untrusted comment:") :].strip(),
        "trusted_comment": None,
        "global_signature": None,
    }
    if len(lines) >= 4 and lines[2].startswith("trusted comment:"):
        out["trusted_comment"] = lines[2][len("trusted comment:") :].strip()
        try:
            gsig = base64.b64decode("".join(lines[3].split()), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise FormatError(f"global signature is not base64: {exc}") from exc
        if len(gsig) != 64:
            raise FormatError(
                f"global signature is {len(gsig)} bytes, expected 64"
            )
        out["global_signature"] = gsig
    return out


def verify_file(artifact_path: str, sig_value: str, pubkey_value: str) -> dict:
    """Verify one artifact. Returns a result dict; never raises for a plain
    verification failure (that is data, not an exception)."""
    result = {
        "artifact": artifact_path,
        "ok": False,
        "reason": None,
        "keyId": None,
        "expectedKeyId": None,
        "alg": None,
        "trustedComment": None,
        "trustedCommentVerified": False,
        "sha256": None,
    }
    key_id, public_key = parse_public_key(pubkey_value)
    result["expectedKeyId"] = key_id[::-1].hex().upper()
    sig = parse_signature(sig_value)
    result["keyId"] = sig["key_id"][::-1].hex().upper()
    result["alg"] = sig["alg"].decode()
    result["trustedComment"] = sig["trusted_comment"]

    if sig["key_id"] != key_id:
        result["reason"] = (
            f"signature was made by key {result['keyId']}, "
            f"not the expected {result['expectedKeyId']}"
        )
        return result

    with open(artifact_path, "rb") as fh:
        data = fh.read()
    result["sha256"] = hashlib.sha256(data).hexdigest()

    message = hashlib.blake2b(data, digest_size=64).digest() if sig["alg"] == b"ED" else data
    if not ed25519_verify(public_key, message, sig["signature"]):
        result["reason"] = "signature does not verify against the expected public key"
        return result

    # The trusted comment is the only part of the container a naive verifier
    # would take on faith. minisign signs it separately; so do we.
    if sig["global_signature"] is not None:
        gmsg = sig["signature"] + (sig["trusted_comment"] or "").encode("utf-8")
        if not ed25519_verify(public_key, gmsg, sig["global_signature"]):
            result["reason"] = "the trusted comment is not covered by a valid signature"
            return result
        result["trustedCommentVerified"] = True

    result["ok"] = True
    return result


# ---------------------------------------------------------------------------
# Self-test: RFC 8032 vectors. The point is to prove this verifier can say NO.
# A verifier that returns True unconditionally passes every "is the release
# signed" check ever written, which is the failure this project has shipped
# twice in other checkers.
# ---------------------------------------------------------------------------

_RFC8032 = [
    # (public key, message, signature) from RFC 8032 §7.1
    (
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
        "",
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
        "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
    ),
    (
        "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
        "72",
        "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da"
        "085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00",
    ),
    (
        "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
        "af82",
        "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac"
        "18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a",
    ),
]


def self_test() -> int:
    failures = []
    for pk_hex, msg_hex, sig_hex in _RFC8032:
        pk, msg, sig = (bytes.fromhex(x) for x in (pk_hex, msg_hex, sig_hex))
        if not ed25519_verify(pk, msg, sig):
            failures.append(f"RFC 8032 vector with message {msg_hex!r} did not verify")
        # Negative control: flip one bit of the message (or use a non-empty
        # message when the vector's message is empty) and require a NO.
        bad_msg = bytes([msg[0] ^ 0x01]) + msg[1:] if msg else b"\x00"
        if ed25519_verify(pk, bad_msg, sig):
            failures.append(f"a tampered message verified for vector {msg_hex!r}")
        bad_sig = bytearray(sig)
        bad_sig[0] ^= 0x01
        if ed25519_verify(pk, msg, bytes(bad_sig)):
            failures.append(f"a tampered signature verified for vector {msg_hex!r}")
        bad_pk = bytearray(pk)
        bad_pk[0] ^= 0x01
        if ed25519_verify(bytes(bad_pk), msg, sig):
            failures.append(f"the wrong public key verified for vector {msg_hex!r}")
    # Non-canonical S must be refused.
    pk, msg, sig = (bytes.fromhex(x) for x in _RFC8032[1][:1] + (_RFC8032[1][1],) + (_RFC8032[1][2],))
    mal = sig[:32] + ((int.from_bytes(sig[32:], "little") + _L) % (1 << 256)).to_bytes(32, "little")
    if ed25519_verify(pk, msg, mal):
        failures.append("a non-canonical S value was accepted")

    for line in failures:
        print(f"  [BAD] {line}")
    if failures:
        print("MINISIGN_VERIFY_SELFTEST_FAIL")
        return 1
    print(
        f"  [ok] {len(_RFC8032)} RFC 8032 vectors verify; tampered message, "
        "tampered signature, wrong key and non-canonical S are all refused"
    )
    print("MINISIGN_VERIFY_SELFTEST_OK")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("artifact", nargs="?", help="the file the signature covers")
    ap.add_argument("--pubkey", help="public key file, base64 blob, or wrapped blob")
    ap.add_argument("--sig", help="signature file (.sig/.minisig) or its content")
    ap.add_argument("--json", action="store_true", help="machine-readable result")
    ap.add_argument("--self-test", action="store_true", help="run the built-in vectors")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not (args.artifact and args.pubkey and args.sig):
        ap.error("need --pubkey, --sig and an artifact (or --self-test)")

    try:
        result = verify_file(args.artifact, args.sig, args.pubkey)
    except FormatError as exc:
        if args.json:
            print(json.dumps({"ok": False, "reason": f"format: {exc}"}))
        else:
            print(f"VERIFY_FORMAT_ERROR: {exc}")
        return 2
    except OSError as exc:
        if args.json:
            print(json.dumps({"ok": False, "reason": f"io: {exc}"}))
        else:
            print(f"VERIFY_IO_ERROR: {exc}")
        return 2

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["ok"]:
        print(
            f"VERIFY_OK key={result['keyId']} alg={result['alg']} "
            f"sha256={result['sha256']}"
        )
        if result["trustedComment"]:
            mark = "verified" if result["trustedCommentVerified"] else "UNSIGNED"
            print(f"  trusted comment ({mark}): {result['trustedComment']}")
    else:
        print(f"VERIFY_FAIL: {result['reason']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
