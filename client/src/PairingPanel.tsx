import { useEffect, useState } from "react";
import { useI18n } from "./i18n";
import {
  accountPairSecret,
  getPairingProvider,
  isValidCode,
  type PairingSnapshot,
  type PeerNode,
  type SelfInfo,
} from "./pairing";
import type { Session } from "./auth";
import { platformGate } from "./platform";
import { useDialog } from "./useDialog";
import { loadSettings } from "./settings";

type View = "choose" | "join" | "active";

function PeerRow(props: { peer: PeerNode; orchestrating: boolean; onMakeCoord: (id: string) => void }) {
  const { t } = useI18n();
  const p = props.peer;
  const hasRange = p.layerLo !== undefined && p.layerHi !== undefined;
  return (
    <div className={`peer${p.role === "coordinator" ? " peer--coord" : ""}`}>
      <span className="peer__dot" />
      <div className="peer__id">
        <span className="peer__host">
          {p.hostname}
          {p.self ? <span className="peer__you"> · {t("pairing.you")}</span> : null}
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
  initialView?: "choose" | "join";
  onSignIn?: () => void;
  onClose: () => void;
}) {
  const { t } = useI18n();
  const dialogRef = useDialog(props.onClose);
  const [view, setView] = useState<View>(props.initialView ?? "choose");
  const [code, setCode] = useState("");
  const [codeErr, setCodeErr] = useState(false);
  const [copied, setCopied] = useState(false);
  // Revealed only on request while running solo; see the active view below.
  const [showCode, setShowCode] = useState(false);
  const [snap, setSnap] = useState<PairingSnapshot | null>(null);

  useEffect(() => {
    const unsub = getPairingProvider().subscribe((s) => {
      setSnap(s);
      if (s.peers.length > 0) setView("active");
    });
    return unsub;
  }, []);

  const create = async () => {
    await getPairingProvider().create(props.self);
  };
  // Account mode (integration plan 3.3): machines signed in to the same
  // platform account derive the same pair secret locally — no code to type.
  // Gate = platform URL configured + a cloud session carrying the user id.
  const gate = platformGate();
  const accountReady = gate.ok && !!gate.session.userId;
  const clusterName = loadSettings().clusterName.trim() || "home";
  const deriveSecret = async (): Promise<string | null> => {
    const g = platformGate();
    if (!g.ok || !g.session.userId) return null;
    return accountPairSecret(g.session.userId, g.url, clusterName);
  };
  const accountCreate = async () => {
    const secret = await deriveSecret();
    if (secret) await getPairingProvider().createAccount(props.self, secret);
  };
  const accountJoin = async () => {
    const secret = await deriveSecret();
    if (secret) await getPairingProvider().joinAccount(props.self, secret);
  };
  const join = async () => {
    if (!isValidCode(code)) {
      setCodeErr(true);
      return;
    }
    setCodeErr(false);
    await getPairingProvider().join(code, props.self);
  };
  const leave = async () => {
    await getPairingProvider().leave();
    setView("choose");
    setCode("");
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
          <button className="iconbtn" onClick={props.onClose} aria-label="✕">
            ✕
          </button>
        </div>

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
            </label>
            <button className="btn-primary btn-block" onClick={join}>
              {t("pairing.join")}
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
                  onMakeCoord={(id) => getPairingProvider().setCoordinator(id)}
                />
              ))}
            </div>
            {snap.phase === "idle" && snap.peers.length < 2 ? (
              <p className="auth-note">{t("pairing.waiting")}</p>
            ) : null}

            {snap.canStart ? (
              <button
                className="btn-primary btn-block"
                onClick={() => getPairingProvider().start()}
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
