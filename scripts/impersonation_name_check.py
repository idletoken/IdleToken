#!/usr/bin/env python3
"""Detect names that impersonate the official project — DIST-13.

The threat: a provider, user or service calling itself "IdleToken Official",
"ldleToken" (lowercase L), "IdIeToken" (capital i), "Idle Token", "ІdleToken"
(Cyrillic І) or "IdleToken.ai Support". Unique-username constraints do not stop
any of these, and React's HTML escaping is about injection, not about two names
that render identically to a human.

The mechanism is Unicode TR39 confusable folding, plus the substitutions people
actually use (leet digits, separators, homoglyph letters). Both are reduced to a
"skeleton"; two names with the same skeleton are visually interchangeable.

WHERE THIS MUST RUN, AND WHERE IT MUST NOT
------------------------------------------
This is a display-name admission check for the SERVER side, at registration and
at rename. Running it in a client would be theatre — the whole premise of the
threat register is that a modified client does whatever it wants (OPS-01).

It is also NOT an authenticator. Passing means "this name is not a visual copy
of ours", not "this account is trustworthy". A provider named `bob` is exactly
as unverified after this check as before it.

Verdicts:
  block   the skeleton IS a protected term — visually the same name
  review  the skeleton contains a protected term, or is one edit away from one
          (e.g. "IdleToken Support", "IdIeTokens") — plausible but not certain
  allow   no relation

Usage:
    impersonation_name_check.py "ІdleToken Official" "bob"
    printf '%s\\n' name1 name2 | impersonation_name_check.py --stdin
    impersonation_name_check.py --self-test
    impersonation_name_check.py --json "some name"

Exit codes: 0 all allow · 1 something was blocked or flagged · 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
CHANNELS = os.path.join(HERE, "release-channels.json")

# Characters that render as an ASCII letter but are not one. This is a hand
# -curated subset of the Unicode confusables table: the Cyrillic and Greek
# lookalikes that make up nearly every real homograph attack, plus the Latin
# variants. NFKC already folds fullwidth, mathematical and circled forms, so
# those do not need entries here.
_CONFUSABLE = {
    # Cyrillic
    "а": "a", "в": "b", "с": "c", "е": "e", "н": "h", "і": "i", "ј": "j",
    "к": "k", "м": "m", "о": "o", "р": "p", "ѕ": "s", "т": "t", "у": "y",
    "х": "x", "ԁ": "d", "ѵ": "v", "ԝ": "w", "ɡ": "g",
    # Greek
    "α": "a", "β": "b", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "ο": "o",
    "ρ": "p", "τ": "t", "υ": "u", "χ": "x", "ζ": "z", "η": "n", "μ": "u",
    # Latin lookalikes and diacritics that survive NFKC
    "ł": "l", "ø": "o", "đ": "d", "ƅ": "b", "ɩ": "l", "ǀ": "l", "ı": "i",
}

# Digit/symbol substitutions, applied AFTER confusable folding.
#
# ⚠ "i" -> "l" is the important one and it is not a typo. In every sans-serif
# UI font, capital I, lowercase l and the digit 1 are the SAME GLYPH. Without
# this fold, "ldleToken" and "IdIeToken" — the two spellings an impersonator
# would actually register — came out only "one edit away" rather than
# identical, which is a review queue instead of a refusal. Folding the whole
# class together is what makes them the same string.
#
# It is safe here because the skeleton is compared only against a handful of
# protected brand terms, never used as a general identity key: the cost of an
# over-broad fold is a false positive on a name that also collapses onto
# "IdleToken", and _MUST_ALLOW in the self-test is what keeps that honest.
_LEET = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t",
         "8": "b", "9": "g", "$": "s", "@": "a", "!": "l", "|": "l",
         "i": "l"}

# Characters that carry no visual weight and are pure obfuscation: zero-width
# joiners, bidi controls, soft hyphens, variation selectors. Anything in
# category Cf, plus a few spacing oddities.
_INVISIBLE_EXTRA = {"­", "͏", "ᅠ", "ㅤ", "ﾠ"}


def _strip_invisible(s: str) -> str:
    return "".join(
        ch for ch in s
        if unicodedata.category(ch) != "Cf" and ch not in _INVISIBLE_EXTRA
    )


def skeleton(name: str, *, aggressive: bool = True) -> str:
    """Reduce a display name to what a human eye actually sees.

    NFKC first (folds fullwidth/mathematical/ligature forms), then invisible
    characters, then confusable letters, then — for the aggressive skeleton —
    leet digits and every separator. Separators go last so that "Idle-Token",
    "idle token" and "IdleToken" all land on the same string.
    """
    s = unicodedata.normalize("NFKC", name)
    s = _strip_invisible(s)
    s = s.casefold()
    s = "".join(_CONFUSABLE.get(ch, ch) for ch in s)
    # Decompose and drop combining marks: "ídlétoken" -> "idletoken".
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    if aggressive:
        s = "".join(_LEET.get(ch, ch) for ch in s)
        s = "".join(ch for ch in s if ch.isalnum())
    else:
        s = "".join(ch if ch.isalnum() else " " for ch in s)
        s = " ".join(s.split())
    return s


def _edit_distance_le1(a: str, b: str) -> bool:
    """True if a and b are at most one insertion/deletion/substitution apart.
    Cheaper and clearer than a full Levenshtein for the only question asked."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(1 for x, y in zip(a, b) if x != y) == 1
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = 0
    while i < la and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def load_protected(path: str = CHANNELS) -> list[str]:
    """The protected terms come from release-channels.json so there is ONE
    list of what counts as the official name — the same file users are pointed
    at. A second hard-coded list here would drift."""
    try:
        with open(path, encoding="utf-8") as fh:
            names = json.load(fh)["brand"]["protectedNames"]
    except Exception:
        names = ["IdleToken"]
    return names


def check(name: str, protected: list[str]) -> dict:
    sk = skeleton(name)
    result = {"name": name, "skeleton": sk, "verdict": "allow", "reason": None,
              "matched": None}
    if not sk:
        result.update(verdict="review", reason="the name has no visible characters")
        return result

    prot = [(p, skeleton(p)) for p in protected]

    for original, psk in prot:
        if sk == psk:
            result.update(verdict="block", matched=original,
                          reason=f"renders identically to the protected name {original!r}")
            return result
    for original, psk in prot:
        # A protected term wholly inside the candidate: "IdleTokenSupport",
        # "OfficialIdleToken". This is the shape real impersonation takes —
        # the copy plus a reassuring word.
        if len(psk) >= 6 and psk in sk:
            result.update(verdict="review", matched=original,
                          reason=f"contains the protected name {original!r}")
            return result
    for original, psk in prot:
        if len(psk) >= 6 and _edit_distance_le1(sk, psk):
            result.update(verdict="review", matched=original,
                          reason=f"one character away from {original!r}")
            return result
    return result


# ---------------------------------------------------------------------------
# Self-test. Both halves matter: a checker that blocks everything is as useless
# as one that blocks nothing, and the false-positive half is what decides
# whether this can be turned on in production at all.
# ---------------------------------------------------------------------------
_MUST_FLAG = [
    ("IdleToken", "block"),                    # the exact name
    ("idletoken", "block"),                    # case
    ("Idle Token", "block"),                   # separator
    ("Idle-Token", "block"),
    ("ldleToken", "block"),                    # lowercase L for capital I
    ("IdIeToken", "block"),                    # capital I for lowercase l
    ("Id1eT0ken", "block"),                    # leet
    ("ＩｄｌｅＴｏｋｅｎ", "block"),                    # fullwidth (NFKC)
    ("Idlе​Token", "block"),              # Cyrillic е + zero-width space
    ("ІdleToken", "block"),                    # Cyrillic І
    ("IdleTokén", "block"),                    # combining acute
    ("IdleToken Official", "review"),          # the classic support-scam shape
    ("Official IdleToken", "review"),
    ("IdleTokens", "review"),                  # one character away
    ("IdleToken Support", "review"),
    ("idletoken.ai", "review"),
]
# Names that must NOT be flagged. If any of these trip, the check cannot be
# enabled — it would block legitimate users, and a check that gets turned off
# protects nobody.
_MUST_ALLOW = [
    "bob", "alice-gpu", "TokenIdle", "Idle", "Token", "idle-gpu-farm",
    "my token server", "IdleHands", "TokenRing", "3090 rig", "闲置算力",
    "Tokenizer", "IdleRPC", "openidle", "tokenfactory",
]


def self_test() -> int:
    protected = load_protected()
    failures = []
    for name, expected in _MUST_FLAG:
        got = check(name, protected)["verdict"]
        # block is strictly stronger than review; accepting it where review was
        # expected is fine, the reverse is not.
        okay = got == expected or (expected == "review" and got == "block")
        if not okay:
            failures.append(f"{name!r}: expected {expected}, got {got}")
    for name in _MUST_ALLOW:
        r = check(name, protected)
        if r["verdict"] != "allow":
            failures.append(
                f"FALSE POSITIVE {name!r}: got {r['verdict']} ({r['reason']})")
    for line in failures:
        print(f"  [BAD] {line}")
    if failures:
        print("IMPERSONATION_CHECK_SELFTEST_FAIL")
        return 1
    print(f"  [ok] {len(_MUST_FLAG)} impersonation shapes are flagged "
          f"(homoglyph, leet, fullwidth, zero-width, combining, separator, suffix)")
    print(f"  [ok] {len(_MUST_ALLOW)} legitimate names are NOT flagged "
          f"(the check can be enabled without blocking real users)")
    print("IMPERSONATION_CHECK_SELFTEST_OK")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--stdin", action="store_true", help="read names from stdin")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    names = list(args.names)
    if args.stdin:
        names += [ln.rstrip("\n") for ln in sys.stdin if ln.strip()]
    if not names:
        ap.error("give at least one name, or --stdin, or --self-test")

    protected = load_protected()
    results = [check(n, protected) for n in names]
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for r in results:
            mark = {"block": "BLOCK ", "review": "REVIEW", "allow": "allow "}[r["verdict"]]
            extra = f"  <- {r['reason']}" if r["reason"] else ""
            print(f"{mark} {r['name']!r}  skeleton={r['skeleton']!r}{extra}")
    return 1 if any(r["verdict"] != "allow" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
