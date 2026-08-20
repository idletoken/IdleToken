import { useEffect, useState } from "react";
import { useI18n } from "./i18n";
import {
  accountPairSecret,
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
import { loadSettings } from "./settings";
import { isSingleNode } from "./models";

type View = "choose" | "join" | "active";

function PeerRow(props: { peer: PeerNode; orchestrating: boolean; onMakeCoord: (id: string) => void }) {
  const { t } = useI18n();
  const p = props.peer;
  const hasRange = p.layerLo !== undefined && p.layerHi !== undefined;
  // Explicit false only: older snapshots (and the dev-sim before the field)
  // omit `online`, and absence has always meant "fine".
  const offline = p.online === false;
  return (
    <div className={`peer${p.role === "coordinator" ? " peer--coord" : ""}${offline ? " peer--offline" : ""}`}>
      <span className="peer__dot" />
      <div className="peer__id">
        <span className="peer__host">
          {p.hostname}
          {p.self ? <span className="peer__you"> · {t("pairing.you")}</span> : null}
          {offline ? <span className="offline-tag">{t("pairing.offline")}</span> : null}
        </span>
        <span className="peer__gpu">
          {hasRange ? t("pairing.layers", { lo: p.layerLo!, hi: p.layerHi! - 1 }) : p.gpu}
        </span>
      </div>
      {props.orchestrating ? (
        <span className={`peer__stage peer__stage--${p.stage}`}>{t(`pairing.stage.${p.stage}` as const)}</span>
      ) : (
        <>
          <span className={`peer__role peer__role--${p.role}`}>
            {p.role === "coordinator" ? t("pairing.coordinator") : t("pairing.worker")}
          </span>
          {p.role !== "coordinator" ? (
            <button className="linkbtn" onClick={() => props.onMakeCoord(p.id)}>
              {t("pairing.makeCoord")}
            </button>
          ) : (
            <span className="peer__spacer" />
          )}
        </>
      )}
    </div>
  );
}

export default function PairingPanel(props: {
  self: SelfInfo;
  // Nullable by design: code-mode pairing needs no account (the engine pairs
  // by code alone) — only the same-account section requires a session.
  session: Session | null;
  // The model this machine is set to serve. A single-node model cannot take a
  // second machine (the coordinator was started with one worker slot and would
  // refuse anyway), so the panel must not offer a join code for one — that
  // would hand out an invitation nobody can accept.
  modelId: string;
  initialView?: "choose" | "join";
  onSignIn?: () => void;
  onClose: () => void;
}) {
  const { t, tErr } = useI18n();
  const dialogRef = useDialog(props.onClose);
  const [view, setView] = useState<View>(props.initialView ?? "choose");
  const [code, setCode] = useState("");
  const [codeErr, setCodeErr] = useState(false);
  const [copied, setCopied] = useState(false);
  // Revealed only on request while running solo; see the active view below.
  const [showCode, setShowCode] = useState(false);
  const [snap, setSnap] = useState<PairingSnapshot | null>(null);
  // Command failures (start refused, coordinator pick refused, …). These come
  // back as rejected invokes; without a catch they were unhandled rejections —
  // the button just did nothing on screen.
  const [opErr, setOpErr] = useState<string | null>(null);
  // Wrap a provider call so its rejection lands on the panel instead of the
  // console. tErr maps client codes to localized copy; unknown text verbatim.
  const guard = (p: Promise<void>) => {
    setOpErr(null);
    return p.catch((e) => setOpErr(tErr(String(e))));
  };

  useEffect(() => {
    const unsub = getPairingProvider().subscribe((s) => {
      setSnap(s);
      if (s.peers.length > 0) setView("active");
      // A background failure tore the roster down (creator's roster port
      // busy, and the like): an "active" view over zero peers would render a
      // ghost cluster — fall back to the entry screen, where the error strip
      // below says what happened.
      else if (s.lastError) setView((v) => (v === "active" ? "choose" : v));
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
      case "subnet":
        return t("pairing.err.subnet");
      case "rejected":
        return t("pairing.err.rejected", { detail: e.detail });
      case "portBusy":
        return t("pairing.err.portBusy", { port: e.detail || "14098" });
      case "creatorLost":
        return t("pairing.err.creatorLost");
      default:
        return e.detail || e.code;
    }
  };

  const create = async () => {
    await guard(getPairingProvider().create(props.self));
  };
  // Account mode (integration plan 3.3): machines signed in to the same
  // platform account derive the same pair secret locally — no code to type.
  // Gate = platform URL configured + a cloud session carrying the user id.
  const gate = platformGate();
  const accountReady = gate.ok && !!gate.session.userId;
  const clusterName = loadSettings().clusterName.trim() || "IdleToken-Home";
  const deriveSecret = async (): Promise<string | null> => {
    const g = platformGate();
    if (!g.ok || !g.session.userId) return null;
    return accountPairSecret(g.session.userId, g.url, clusterName);
  };
  const accountCreate = async () => {
    const secret = await deriveSecret();
    if (secret) await guard(getPairingProvider().createAccount(props.self, secret));
  };
  const accountJoin = async () => {
    const secret = await deriveSecret();
    if (secret) await guard(getPairingProvider().joinAccount(props.self, secret));
  };
  const join = async () => {
    if (!isValidCode(code)) {
      setCodeErr(true);
      return;
    }
    setCodeErr(false);
    await guard(getPairingProvider().join(code, props.self));
  };
  const leave = async () => {
    await guard(getPairingProvider().leave());
    setView("choose");
    setCode("");
  };
  // "Running on one machine", not merely "one peer": while the cluster is still
  // forming, the roster is legitimately one machine and the code must stay put.
  const soloRunning = !!snap && snap.phase !== "idle" && snap.peers.length === 1;
  const clusterable = !isSingleNode(props.modelId);
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
          <p className="field__hint field__hint--err" role="alert">
            {lastErrText(snap.lastError)}
          </p>
        ) : null}

        {view === "choose" ? (
          <>
            <button className="choice" onClick={create}>
              <span className="choice__title">{t("pairing.chooseCreate")}</span>
              <span className="choice__hint">{t("pairing.chooseCreateHint")}</span>
            </button>
            <button className="choice" onClick={() => setView("join")}>
              <span className="choice__title">{t("pairing.chooseJoin")}</span>
              <span className="choice__hint">{t("pairing.chooseJoinHint")}</span>
            </button>
            {accountReady ? (
              <>
                <div className="setting-group__label" style={{ marginTop: 8 }}>
                  {t("pairing.accountTitle")}
                </div>
                <button className="choice" onClick={accountCreate}>
                  <span className="choice__title">{t("pairing.accountCreate")}</span>
                  <span className="choice__hint">{t("pairing.accountCreateHint")}</span>
                </button>
                <button className="choice" onClick={accountJoin}>
                  <span className="choice__title">{t("pairing.accountJoin")}</span>
                  <span className="choice__hint">
                    {t("pairing.accountJoinHint", { email: props.session?.email ?? "" })}
                  </span>
                </button>
                <p className="auth-note">{t("pairing.accountLanHint", { name: clusterName })}</p>
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
            <button className="btn-primary btn-block" onClick={join}>
              {snap?.lastError ? t("state.retry") : t("pairing.join")}
            </button>
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
            {/* Running solo on a single-node model: there is no "add a machine"
                to offer. Say why once, instead of a button that leads to a code
                the coordinator will not honour. */}
            {soloRunning && !clusterable ? (
              <p className="auth-note">{t("pairing.singleNodeModel")}</p>
            ) : null}
            {snap.code && soloRunning && clusterable && !showCode ? (
              <button className="linkbtn linkbtn--center" onClick={() => setShowCode(true)}>
                {t("pairing.addMachine")}
              </button>
            ) : null}
            {snap.code && (!soloRunning || showCode) && (clusterable || !soloRunning) ? (
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
              <div className="phase-banner">
                <span className="spinner spinner--sm" />
                {t(`pairing.phase.${snap.phase}` as const)}
              </div>
            ) : null}

            <div className="setting-group__label" style={{ marginTop: 4 }}>
              {t("pairing.members", { n: snap.peers.length })}
            </div>
            <div className="peer-list">
              {snap.peers.map((p) => (
                <PeerRow
                  key={p.id}
                  peer={p}
                  orchestrating={snap.phase !== "idle"}
                  onMakeCoord={(id) => void guard(getPairingProvider().setCoordinator(id))}
                />
              ))}
            </div>
            {snap.phase === "idle" && snap.peers.length < 2 ? (
              <p className="auth-note">{t("pairing.waiting")}</p>
            ) : null}

            {snap.canStart ? (
              <button
                className="btn-primary btn-block"
                onClick={() => void guard(getPairingProvider().start())}
              >
                {t("pairing.startCluster", { n: snap.peers.length })} →
              </button>
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
