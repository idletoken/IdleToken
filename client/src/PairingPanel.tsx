import { useEffect, useRef, useState } from "react";
import { useI18n } from "./i18n";
import {
  approveCoordinatorRequest,
  accountPairSecret,
  canApproveCoordinatorRequest,
  COORDINATOR_ROLE_RISK,
  getPairingProvider,
  isValidCode,
  type PairingError,
  type PairingSnapshot,
  type PeerNode,
  type SelfInfo,
} from "./pairing";
import type { Session } from "./auth";
import { platformGate } from "./platform";
import { useDialog } from "./useDialog";
import { loadSettings, overflowTuning, type EngineTuning } from "./settings";
import { ctxLabel, floorGiB1, fmtQuant } from "./format";
import { isMoeModel } from "./models";
import StartupProgress from "./StartupProgress";

type View = "choose" | "join" | "active";

function PeerRow(props: {
  peer: PeerNode;
  snapshot: PairingSnapshot;
  orchestrating: boolean;
  onApproveCoordinator: (id: string) => void;
}) {
  const { t } = useI18n();
  const p = props.peer;
  const showMoeResources = isMoeModel(props.snapshot.modelId ?? "");
  const mem = (bytes?: number) => bytes ? `${floorGiB1(bytes)} GB` : "—";
  const hasRange = p.layerLo !== undefined && p.layerHi !== undefined;
  // Explicit false only: older snapshots (and the dev-sim before the field)
  // omit `online`, and absence has always meant "fine".
  const offline = p.online === false;
  const creatorCanSeeRequest = props.snapshot.isCreator === true
    && p.self === false
    && p.role !== "coordinator"
    && p.online !== false
    && p.wantsCoordinator === true;
  const canApprove = canApproveCoordinatorRequest(props.snapshot, p);
  return (
    <div>
      <div className={`peer${p.role === "coordinator" ? " peer--coord" : ""}${offline ? " peer--offline" : ""}`}>
        <span className="peer__dot" />
        <div className="peer__id">
          <span className="peer__host">
            {p.hostname}
            {p.self ? <span className="peer__you"> · {t("pairing.you")}</span> : null}
            {offline ? <span className="offline-tag">{t("pairing.offline")}</span> : null}
          </span>
          <span className="peer__gpu">
            {hasRange
              ? t("pairing.layers", { lo: p.layerLo!, hi: p.layerHi! - 1 })
              : p.modelReady === false
                ? t("pairing.model.preparing")
                : p.gpu}
          </span>
          {showMoeResources ? (
            <span className="peer__resources">
              {t("pairing.moeResources", {
                gpu: mem(p.vramFree),
                // Show the schedulable expert pool, not ordinary free RAM.
                // On Windows this has already paid the per-node WDDM reserve.
                ram: mem(p.ramExpertFree),
              })}
              {p.unifiedMemory ? ` · ${t("node.unified")}` : ""}
            </span>
          ) : null}
        </div>
        {props.orchestrating ? (
          <span className={`peer__stage peer__stage--${p.stage}`}>{t(`pairing.stage.${p.stage}` as const)}</span>
        ) : (
          <>
            <span className={`peer__role peer__role--${p.role}`}>
              {p.role === "coordinator" ? t("pairing.coordinator") : t("pairing.worker")}
            </span>
            <span className="peer__spacer" />
          </>
        )}
      </div>
      {creatorCanSeeRequest ? (
        <div className="auth-note" role="status" style={{ margin: "4px 12px 12px" }}>
          <strong>{p.hostname} requests the coordinator role.</strong>{" "}
          Device identity: <code>{p.deviceIdentity || "unavailable — update this machine"}</code>.
          <br />
          {COORDINATOR_ROLE_RISK}
          <br />
          {canApprove ? (
            <button className="linkbtn" onClick={() => props.onApproveCoordinator(p.id)}>
              Review and approve…
            </button>
          ) : (
            <span> Approval is blocked until a stable device identity is available.</span>
          )}
        </div>
      ) : null}
    </div>
  );
}

export default function PairingPanel(props: {
  self: SelfInfo;
  // Nullable by design: code-mode pairing needs no account (the engine pairs
  // by code alone) — only the same-account section requires a session.
  session: Session | null;
  initialView?: "choose" | "join";
  // Preflight for BOTH roles: resolve and integrity-check the locally selected
  // GGUF, downloading nothing. Returns the exact local primary GGUF path, or
  // throws `[WEIGHTS_NOT_DOWNLOADED]` when the parts are not all here. A
  // creator may not advertise a cluster without it; since 2026-09-01 a joiner
  // may not be admitted without it either.
  prepareSelectedModel: () => Promise<string>;
  /** Fetch and verify the exact model a cluster demanded, then hand back the
   *  path and the tuning that names it. Both are returned rather than read back
   *  from props because the join follows in the same turn (see `join`). */
  prepareClusterModel: (modelId: string, quant: string) => Promise<{
    modelPath: string;
    tuning: EngineTuning;
  }>;
  onSignIn?: () => void;
  onClose: () => void;
}) {
  const { t, tErr } = useI18n();
  const dialogRef = useDialog(props.onClose);
  const [view, setView] = useState<View>(props.initialView ?? "choose");
  const [code, setCode] = useState("");
  const [codeErr, setCodeErr] = useState(false);
  const [copied, setCopied] = useState(false);
  // A weights fetch started from the refusal card is running.
  const [fetching, setFetching] = useState(false);
  // Which kind of join produced the current refusal. A modelNotReady card has
  // to retry the SAME kind, and the two are reachable from the same screen
  // (code-mode "back", then the account section) — so this cannot be inferred
  // from the view without occasionally retrying the wrong one.
  const [joinKind, setJoinKind] = useState<"code" | "account" | null>(null);
  // Revealed only on request while running solo; see the active view below.
  const [showCode, setShowCode] = useState(false);
  const [snap, setSnap] = useState<PairingSnapshot | null>(null);
  // All four entry choices operate this machine as a compute node. They share
  // the same local-weight prerequisite; joining is not an escape hatch around
  // the download gate. The action handlers repeat the preflight below so a
  // file removed after render cannot bypass a disabled-button check.
  const modelReady = !!props.self.modelPath;
  // Command failures (start refused, coordinator pick refused, …). These come
  // back as rejected invokes; without a catch they were unhandled rejections —
  // the button just did nothing on screen.
  const [opErr, setOpErr] = useState<string | null>(null);
  // A join command returns after starting LAN discovery; the real result
  // arrives later in a snapshot. Keep the action visibly alive across that
  // gap instead of letting the button look as though it did nothing.
  const [connecting, setConnecting] = useState<"code" | "account" | null>(null);
  const [startRequested, setStartRequested] = useState(false);
  // A panel opened from "Manage cluster" must stay open, including when its
  // first snapshot is already ready. Only an action started inside this panel
  // arms auto-close: a join closes once this machine appears in the roster;
  // Start closes once orchestration has actually left the idle phase. The
  // dashboard then owns the progress UI and the management dialog no longer
  // sits over it after a successful connection.
  const autoClose = useRef<"joined" | "started" | null>(null);
  // Wrap a provider call so its rejection lands on the panel instead of the
  // console. tErr maps client codes to localized copy; unknown text verbatim.
  const guard = (p: Promise<void>, onError?: () => void) => {
    setOpErr(null);
    return p.catch((e) => {
      onError?.();
      setOpErr(tErr(String(e)));
    });
  };

  useEffect(() => {
    const unsub = getPairingProvider().subscribe((s) => {
      setSnap(s);
      if (s.peers.length > 0) {
        setConnecting(null);
        if (s.phase !== "idle") setStartRequested(false);
        setView("active");
        if (
          autoClose.current === "joined"
          || (autoClose.current === "started" && s.phase !== "idle")
        ) {
          autoClose.current = null;
          props.onClose();
        }
      }
      // A background failure tore the roster down (creator's roster port
      // busy, and the like): an "active" view over zero peers would render a
      // ghost cluster — fall back to the entry screen, where the error strip
      // below says what happened.
      else if (s.lastError) {
        autoClose.current = null;
        setConnecting(null);
        setStartRequested(false);
        setView((v) => (v === "active" ? "choose" : v));
      }
    });
    return unsub;
  }, []);

  // Async pairing failures (no cluster found / rejected / creator's roster
  // port busy) arrive through the snapshot, well after the invoke resolved —
  // render them where the user acted, with the concrete things to check.
  const lastErrText = (e: PairingError): string => {
    switch (e.code) {
      case "notFound":
        return t("pairing.err.notFound", { port: e.detail || "14099" });
      case "notFoundManual":
        return t("pairing.err.notFoundManual");
      case "badCode":
        return t("pairing.err.badCode");
      case "oldCreator":
        return t("pairing.err.oldCreator");
      case "subnet":
        return t("pairing.err.subnet");
      // The Rust-side refusal code; the message key is `joinNeedsModel` because
      // `pairing.err.modelNotReady` already names a different failure (a member
      // that is in the cluster but not finished preparing, raised at start).
      case "modelNotReady": {
        // One wire code, two causes (2026-09-02): the weights are missing, or
        // the context window disagrees. The creator does not distinguish them —
        // it only reports what it requires — so decide here, where this
        // machine's own setting is known. Telling someone to download weights
        // they already have is how a one-click fix becomes an hour.
        const need = snap?.requiredModel;
        const localCtx = loadSettings().ctxTokens;
        if (need?.ctx && need.ctx !== localCtx) {
          return t("pairing.err.joinNeedsCtx", {
            want: ctxLabel(need.ctx),
            have: ctxLabel(localCtx),
          });
        }
        return t("pairing.err.joinNeedsModel", {
          model: need?.modelId ?? "?",
          quant: need?.quant ?? "?",
        });
      }
      case "rejected":
        return t("pairing.err.rejected", { detail: e.detail });
      case "portBusy":
        return t("pairing.err.portBusy", { port: e.detail || "14098" });
      case "creatorLost":
        return t("pairing.err.creatorLost");
      case "lanRoute":
        return t("pairing.err.lanRoute", { detail: e.detail });
      default:
        return e.detail || e.code;
    }
  };

  const create = async () => {
    await guard(
      props.prepareSelectedModel().then((modelPath) =>
        getPairingProvider().create({ ...props.self, modelPath })
      )
    );
  };
  // Account mode (integration plan 3.3): machines signed in to the same
  // platform account derive the same pair secret locally — no code to type.
  // Gate = platform URL configured + a cloud session carrying the user id.
  const gate = platformGate();
  const accountReady = gate.ok && !!gate.session.userId;
  const deriveSecret = async (): Promise<string | null> => {
    const g = platformGate();
    if (!g.ok || !g.session.userId) return null;
    return accountPairSecret(g.session.userId, g.url);
  };
  const accountCreate = async () => {
    const secret = await deriveSecret();
    if (secret) {
      await guard(
        props.prepareSelectedModel().then((modelPath) =>
          getPairingProvider().createAccount({ ...props.self, modelPath }, secret)
        )
      );
    }
  };
  const accountJoin = async (over?: { modelPath: string; tuning: EngineTuning }) => {
    const secret = await deriveSecret();
    if (!secret) return;
    setJoinKind("account");
    setConnecting("account");
    autoClose.current = "joined";
    await guard(
      (async () => {
        const modelPath = over?.modelPath ?? (await props.prepareSelectedModel());
        await getPairingProvider().joinAccount(
          { ...props.self, modelPath, tuning: over?.tuning ?? props.self.tuning },
          secret
        );
      })(),
      () => {
        autoClose.current = null;
        setConnecting(null);
      },
    );
  };

  // `over` carries the weights this machine has JUST fetched after a
  // modelNotReady refusal. It has to be threaded through rather than read from
  // props: the download saves new settings and the join goes out in the same
  // turn, before React has committed them, so `props.self` still describes the
  // model we were refused for. Sending that would earn a second refusal for the
  // model we just spent an hour downloading.
  const join = async (over?: { modelPath: string; tuning: EngineTuning }) => {
    if (!isValidCode(code)) {
      setCodeErr(true);
      return;
    }
    setCodeErr(false);
    setJoinKind("code");
    setConnecting("code");
    autoClose.current = "joined";
    await guard(
      (async () => {
        // Admission requires verified local weights. Never send an empty path:
        // the four entry choices are disabled while the selected model is
        // missing, and this preflight closes the file-removal/race window.
        const modelPath = over?.modelPath ?? (await props.prepareSelectedModel());
        await getPairingProvider().join(code, {
          ...props.self,
          modelPath,
          tuning: over?.tuning ?? props.self.tuning,
        });
      })(),
      () => {
        autoClose.current = null;
        setConnecting(null);
      },
    );
  };

  // The one recovery path from a modelNotReady refusal: fetch exactly what the
  // cluster named, then retry. Never automatic — this is a download the size of
  // the model, so it happens when the user asks for it and not before.
  const fetchRequiredAndJoin = async () => {
    const need = snap?.requiredModel;
    if (!need || fetching) return;
    setFetching(true);
    setOpErr(null);
    try {
      const over = await props.prepareClusterModel(need.modelId, need.quant);
      await (joinKind === "account" ? accountJoin(over) : join(over));
    } catch (e) {
      setOpErr(tErr(String(e)));
    } finally {
      setFetching(false);
    }
  };
  const leave = async () => {
    setOpErr(null);
    try {
      await getPairingProvider().leave();
      // Leaving finishes this task. Keeping the dialog open and switching it
      // back to `choose` made four create/join choices appear as if leaving a
      // cluster immediately required forming another one.
      props.onClose();
    } catch (e) {
      setOpErr(tErr(String(e)));
    }
  };
  const approveCoordinator = (peerId: string) => {
    if (!snap) return;
    void guard(
      approveCoordinatorRequest(
        getPairingProvider(),
        snap,
        peerId,
        (message) => window.confirm(message),
      ).then(() => undefined),
    );
  };
  // "Running on one machine", not merely "one peer": while the cluster is still
  // forming, the roster is legitimately one machine and the code must stay put.
  const soloRunning = !!snap && snap.phase !== "idle" && snap.peers.length === 1;
  const copyCode = async () => {
    if (!snap?.code) return;
    try {
      await navigator.clipboard.writeText(snap.code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard may be blocked; the code is shown regardless */
    }
  };
  return (
    <div className="modal-scrim" onClick={props.onClose}>
      <div ref={dialogRef} className="modal modal--auth" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
        <div className="modal__head">
          <div>
            <h2>{t("pairing.title")}</h2>
            {snap?.source === "dev-sim" ? (
              <span className="devsim-badge" title={t("pairing.devSimNote")}>
                {t("pairing.devSim")}
              </span>
            ) : null}
          </div>
          <button className="iconbtn" onClick={props.onClose} aria-label={t("a11y.close")}>
            ✕
          </button>
        </div>

        {/* Command failures (start refused, and the like) — one shared strip,
            wherever in the flow they happen. */}
        {opErr ? <p className="field__hint field__hint--err">{opErr}</p> : null}
        {/* Background pairing failures outside the join form (the join view
            renders lastError next to the code input instead). */}
        {view !== "join" && snap?.lastError ? (
          <>
            <p className="field__hint field__hint--err" role="alert">
              {lastErrText(snap.lastError)}
            </p>
            {/* An account-mode join lands here rather than in the join view, so
                its modelNotReady refusal needs the same one-click recovery. */}
            {snap.lastError.code === "modelNotReady" && snap.requiredModel ? (
              <button
                className="btn-primary btn-block"
                disabled={fetching}
                onClick={() => void fetchRequiredAndJoin()}
              >
                {fetching
                  ? t("pairing.fetchingModel")
                  : t("pairing.fetchModel", {
                      model: snap.requiredModel.modelId,
                      quant: fmtQuant(snap.requiredModel.quant),
                    })}
              </button>
            ) : null}
          </>
        ) : null}

        {view === "choose" ? (
          <>
            {connecting === "account" ? (
              <StartupProgress
                compact
                label={t("startup.finding")}
                detail={t("startup.findingDetail")}
              />
            ) : null}
            <button className="choice" onClick={create} disabled={!modelReady || connecting !== null}>
              <span className="choice__title">{t("pairing.chooseCreate")}</span>
              <span className="choice__hint">
                {modelReady ? t("pairing.chooseCreateHint") : t("pairing.needsModel")}
              </span>
            </button>
            <button className="choice" onClick={() => setView("join")} disabled={!modelReady || connecting !== null}>
              <span className="choice__title">{t("pairing.chooseJoin")}</span>
              <span className="choice__hint">
                {modelReady ? t("pairing.chooseJoinHint") : t("pairing.needsModel")}
              </span>
            </button>
            {accountReady ? (
              <>
                <div className="setting-group__label" style={{ marginTop: 8 }}>
                  {t("pairing.accountTitle")}
                </div>
                <button className="choice" onClick={accountCreate} disabled={!modelReady || connecting !== null}>
                  <span className="choice__title">{t("pairing.accountCreate")}</span>
                  <span className="choice__hint">
                    {modelReady ? t("pairing.accountCreateHint") : t("pairing.needsModel")}
                  </span>
                </button>
                {/* Wrapped: accountJoin now takes an optional "weights we just
                    fetched" argument, and a bare handler would hand it the
                    click event as that argument. */}
                <button className="choice" onClick={() => void accountJoin()} disabled={!modelReady || connecting !== null}>
                  <span className="choice__title">{t("pairing.accountJoin")}</span>
                  <span className="choice__hint">
                    {modelReady
                      ? t("pairing.accountJoinHint", { email: props.session?.email ?? "" })
                      : t("pairing.needsModel")}
                  </span>
                </button>
              </>
            ) : (
              <p className="auth-note">
                {t("pairing.accountNeedLogin")}
                {!props.session && props.onSignIn ? (
                  <>
                    {" "}
                    <button className="linkbtn" onClick={props.onSignIn}>
                      {t("auth.submitSignIn")}
                    </button>
                  </>
                ) : null}
              </p>
            )}
          </>
        ) : null}

        {view === "join" ? (
          <>
            <label className="field">
              <span className="field__k">{t("pairing.enterCode")}</span>
              <input
                className="field__input code-input"
                disabled={!modelReady}
                value={code}
                maxLength={6}
                autoCapitalize="characters"
                placeholder="K7QP2M"
                onChange={(e) => {
                  setCode(e.target.value.toUpperCase());
                  setCodeErr(false);
                }}
              />
              {codeErr ? <span className="field__hint field__hint--err">{t("pairing.invalidCode")}</span> : null}
              {/* The join runs in the background after the invoke returns, so
                  its failure (no cluster found, code rejected) arrives via the
                  snapshot — render it here, red, with the retry button right
                  below. It used to go only to stderr while the panel silently
                  reset, which looked like the button doing nothing. */}
              {snap?.lastError ? (
                <span className="field__hint field__hint--err" role="alert">
                  {lastErrText(snap.lastError)}
                </span>
              ) : null}
            </label>
            {/* A modelNotReady refusal is the one failure retrying cannot fix:
                this machine does not have the cluster's weights, and pressing
                Join again only earns the same refusal. Offer the exact fetch
                instead — named, so the user is agreeing to a specific download
                and not to "whatever the cluster wants". */}
            {snap?.lastError?.code === "modelNotReady" && snap.requiredModel ? (
              <button
                className="btn-primary btn-block"
                disabled={fetching}
                onClick={() => void fetchRequiredAndJoin()}
              >
                {fetching
                  ? t("pairing.fetchingModel")
                  : t("pairing.fetchModel", {
                      model: snap.requiredModel.modelId,
                      quant: fmtQuant(snap.requiredModel.quant),
                    })}
              </button>
            ) : (
              <>
                {connecting === "code" ? (
                  <StartupProgress
                    compact
                    label={t("startup.finding")}
                    detail={t("startup.findingDetail")}
                  />
                ) : null}
                <button
                  className="btn-primary btn-block"
                  disabled={!modelReady || connecting !== null}
                  onClick={() => void join()}
                >
                  {connecting === "code"
                    ? t("startup.finding")
                    : snap?.lastError
                      ? t("state.retry")
                      : t("pairing.join")}
                </button>
              </>
            )}
            <button className="linkbtn linkbtn--center" onClick={() => setView("choose")}>
              {t("pairing.back")}
            </button>
          </>
        ) : null}

        {view === "active" && snap ? (
          <>
            {/* A RUNNING one-machine deployment does not lead with a join code.
                Someone who picked "just run here" has nobody to share it with,
                and a 6-character secret sitting at the top of the panel reads
                like a step they still owe.

                Not deleted, though: the standalone empty state promises "you can
                add machines later from Manage", and this code is the only way to
                do that — removing it outright would make that sentence false. It
                moves behind an explicit ask instead.

                Only when RUNNING: while the cluster is still forming (idle) the
                code is the whole point of the screen, however many peers. */}
            {snap.code && soloRunning && !showCode ? (
              <button className="linkbtn linkbtn--center" onClick={() => setShowCode(true)}>
                {t("pairing.addMachine")}
              </button>
            ) : null}
            {snap.code && (!soloRunning || showCode) ? (
              <div className="code-share">
                <span className="code-share__label">{t("pairing.yourCode")}</span>
                <div className="code-share__row">
                  <span className="code-share__code">{snap.code}</span>
                  <button className="iconbtn" onClick={copyCode}>
                    {copied ? t("pairing.copied") : t("pairing.copy")}
                  </button>
                </div>
              </div>
            ) : null}
            {snap.accountMode ? (
              <p className="auth-note">
                {t("pairing.accountCluster", { email: props.session?.email ?? "" })}
              </p>
            ) : null}

            {snap.phase !== "idle" && snap.phase !== "ready" ? (
              <StartupProgress
                compact
                label={t(`pairing.phase.${snap.phase}` as const)}
                detail={snap.peers.length > 1 ? t("startup.clusterDetail") : t("startup.localDetail")}
              />
            ) : null}

            <div className="setting-group__label" style={{ marginTop: 4 }}>
              {t("pairing.members", { n: snap.peers.length })}
            </div>
            <div className="peer-list">
              {snap.peers.map((p) => (
                <PeerRow
                  key={p.id}
                  peer={p}
                  snapshot={snap}
                  orchestrating={snap.phase !== "idle"}
                  onApproveCoordinator={approveCoordinator}
                />
              ))}
            </div>
            {snap.phase === "idle" && snap.peers.length < 2 ? (
              <p className="auth-note">{t("pairing.waiting")}</p>
            ) : null}

            {snap.phase === "idle" && snap.peers.some((p) => p.modelReady === false) ? (
              <p className="auth-note">{t("pairing.model.waiting")}</p>
            ) : null}

            {snap.canStart ? (
              <div className="cluster-start-actions cluster-start-actions--modal">
                <button
                  className="btn-primary btn-block cluster-start"
                  disabled={startRequested}
                  onClick={() => {
                    setStartRequested(true);
                    autoClose.current = "started";
                    void guard(
                      getPairingProvider().start(
                        false,
                        props.self.modelPath,
                        overflowTuning(loadSettings()),
                      ),
                      () => {
                        autoClose.current = null;
                        setStartRequested(false);
                      },
                    );
                  }}
                >
                  {startRequested
                    ? t("pairing.phase.starting")
                    : `${t("pairing.startCluster", { n: snap.peers.length })} →`}
                </button>
              </div>
            ) : null}

            {/* The API address lives on the dashboard's cluster card now —
                this panel stays a management surface (members / code / leave). */}
            <div className="modal__foot">
              <button className="linkbtn" onClick={leave}>
                {t("pairing.leave")}
              </button>
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}
