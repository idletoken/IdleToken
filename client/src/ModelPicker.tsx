// The model picker, as a popover you can open from anywhere the current model
// is displayed (the chat header, the cluster card).
//
// Why it is not just a dropdown that writes the setting: a model is chosen when
// the coordinator LOADS it — there is no hot swap in the engine — so switching
// while something is running means stopping it and building it again. That is
// a real consequence and the picker states it before doing it:
//
//   nothing running  -> the pick is the whole operation, applied silently
//   one machine      -> restarts this machine's engine (and downloads the
//                       weights first if they are not here yet)
//   several machines -> restarts the CLUSTER: this machine goes back to the
//                       roster screen holding the same join code, and the other
//                       machines rejoin it
//
// The alternative — let the dropdown change a setting that does nothing until
// the next restart — is worse than a link to Settings: it looks like it worked.
//
// The list is the CURATED registry, and it is the whole list: the open intake
// (pick any GGUF file / HF repo) was removed on 2026-08-15 — serving arbitrary
// models is not this product's job, and shared endpoints must run models whose
// quality we can vouch for. Model requests go through GitHub issues.
import { useEffect, useState } from "react";
import { useI18n } from "./i18n";
import { AVAILABLE_MODELS, defaultQuant, hasQuantChoice, quantOptions, isSingleNode } from "./models";
import { fmtBytes } from "./format";
import { useDialog } from "./useDialog";
import { loadCapability, type CapabilityMode, type CapabilityRow } from "./Capability";
import { fetchLeaderboard } from "./platform";
import { openExternal } from "./auth";
import { ISSUES_URL } from "./links";

/** What the cluster is running right now, which is what a switch would have to
 *  restart. `null` = nothing is running, so a pick costs nothing.
 *
 *  `modelId`/`quant` may be empty when an older engine reports no model: then
 *  every pick counts as a change and asks for confirmation, which is the safe
 *  direction — the alternative is restarting a cluster without saying so. */
export interface RunningModel {
  modelId: string;
  quant: string;
  /** Machines in the running cluster, this one included. */
  machines: number;
}

/** The advisor's three states, worded exactly as the capability table words
 *  them — one verdict must not read differently on two screens. "unavailable"
 *  (an old engine's "backend not implemented") collapses into "won't run",
 *  same as it does there. */
function fitKey(mode: CapabilityMode): "cap.yesGpu" | "cap.yesHybrid" | "cap.no" {
  if (mode === "gpu_only") return "cap.yesGpu";
  if (mode === "hybrid") return "cap.yesHybrid";
  return "cap.no";
}
function fitClass(mode: CapabilityMode): string {
  if (mode === "gpu_only") return "is-fast";
  if (mode === "hybrid") return "is-slow";
  return "is-no";
}

/** 1_234_567 → "1.2M". Order of magnitude is the point; exact counts are not. */
function shortCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(n >= 1e4 ? 0 : 1)}K`;
  return String(n);
}

export default function ModelPicker(props: {
  /** The local setting: what the next start would use. */
  modelId: string;
  quant: string;
  running: RunningModel | null;
  /** The paired cluster's API, when there is one: the fit verdicts then cover
   *  the whole pool instead of this machine alone. */
  apiBaseUrl?: string | null;
  /** Apply the pick. The caller saves it and performs any restart. */
  onPick: (modelId: string, quant: string) => void;
  onClose: () => void;
}) {
  const { t } = useI18n();
  const ref = useDialog(props.onClose);

  // Ordering by real usage on the platform (7-day window). Enhancement only:
  // no platform configured, offline, slow, or too little traffic to rank
  // (`source: 'seeded'`) all end the same way — the static manifest order, no
  // label, no numbers. That degradation is deliberate and it is VISIBLE: the
  // "ordered by usage" line is absent, so nobody reads the list as a ranking
  // it is not. Faking either the order or the numbers is the thing not allowed.
  const [usage, setUsage] = useState<Map<string, number> | null>(null);
  useEffect(() => {
    let alive = true;
    fetchLeaderboard("week")
      .then((p) => {
        if (!alive || p.source !== "measured") return;
        setUsage(new Map(p.rows.map((r) => [r.model, r.totalTokens])));
      })
      .catch(() => {
        /* enhancement only — the manifest order is a fine answer */
      });
    return () => {
      alive = false;
    };
  }, []);

  // The three-state fit verdict per model. NOT recomputed here: it comes from
  // the engine's advisor through the same loader the capability table uses
  // (Capability.tsx). A second derivation in TypeScript is exactly what that
  // file's header forbids.
  const [fit, setFit] = useState<Map<string, CapabilityRow[]>>(new Map());
  useEffect(() => {
    let alive = true;
    loadCapability(props.apiBaseUrl)
      .then((rep) => {
        if (!alive) return;
        const by = new Map<string, CapabilityRow[]>();
        for (const r of rep.models) by.set(r.id, [...(by.get(r.id) ?? []), r]);
        setFit(by);
      })
      .catch(() => {
        /* no advisor (plain browser, engine down) → no pills, no error box */
      });
    return () => {
      alive = false;
    };
  }, [props.apiBaseUrl]);
  // A pick that is waiting for confirmation because it would restart something.
  const [pending, setPending] = useState<{ modelId: string; quant: string } | null>(null);

  // Popover, not a modal: clicking anywhere else dismisses it. useDialog gives
  // Escape and the focus trap; there is no scrim to click, so outside-click is
  // this component's job. Mousedown (not click) so it closes on the press, and
  // deferred one tick so the very press that OPENED it does not close it.
  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) props.onClose();
    };
    const timer = setTimeout(() => document.addEventListener("mousedown", onDown), 0);
    return () => {
      clearTimeout(timer);
      document.removeEventListener("mousedown", onDown);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const run = props.running;
  // What the chip/row shows as current: the running model when there is one,
  // otherwise the setting.
  const cur = run && run.modelId
    ? { modelId: run.modelId, quant: run.quant }
    : { modelId: props.modelId, quant: props.quant || defaultQuant(props.modelId) };

  const choose = (modelId: string, quant: string) => {
    const sameAsSetting = modelId === props.modelId && quant === (props.quant || defaultQuant(props.modelId));
    // Picking what is ALREADY RUNNING never restarts anything, even when the
    // local setting says something else — in that case the pick is how you
    // resolve the disagreement (it saves the running model as your setting),
    // and making someone confirm a restart to keep what they already have
    // would be nonsense.
    const sameAsRunning =
      !!run && !!run.modelId && modelId === run.modelId && (!run.quant || !quant || quant === run.quant);
    if (sameAsSetting && (!run || sameAsRunning)) {
      props.onClose();
      return;
    }
    if (!run || sameAsRunning) {
      props.onPick(modelId, quant);
      props.onClose();
      return;
    }
    setPending({ modelId, quant });
  };

  const confirm = () => {
    if (!pending) return;
    props.onPick(pending.modelId, pending.quant);
    props.onClose();
  };

  const curQuant = cur.quant || defaultQuant(cur.modelId);

  // Usage order when we have it, manifest order otherwise. Models the platform
  // has never served keep their manifest position AFTER the ranked ones — a
  // model nobody has run yet is not the same as a model that ranked last.
  const ordered = usage
    ? [...AVAILABLE_MODELS].sort((a, b) => (usage.get(b.id) ?? -1) - (usage.get(a.id) ?? -1))
    : AVAILABLE_MODELS;

  /** The advisor's verdict for this model at this precision (exact row first,
   *  otherwise any row for the model — an older engine may report one quant). */
  const fitOf = (modelId: string, quant: string): CapabilityMode | null => {
    const rows = fit.get(modelId);
    if (!rows || rows.length === 0) return null;
    return (rows.find((r) => r.quant === quant) ?? rows[0]).mode;
  };

  // The DRAFT pick (2026-08-15). Clicking a row used to apply immediately —
  // with a cluster running, the restart confirmation popped before the user
  // could even reach the precision dropdown. Now the row and the dropdown
  // only edit this draft; the Apply button is the one thing that acts.
  const [sel, setSel] = useState({ modelId: cur.modelId, quant: curQuant });

  return (
    <div className="modelpick" ref={ref} role="dialog" aria-label={t("model.pick.title")}>
      {pending ? (
        <div className="modelpick__confirm">
          <p className="modelpick__confirm-title">
            {t("model.switch.title", {
              model: AVAILABLE_MODELS.find((m) => m.id === pending.modelId)?.label ?? pending.modelId,
            })}
          </p>
          {/* One machine and several machines are different promises. The
              second one is about OTHER people's machines, so it says how many
              and what they will have to do. */}
          <p className="modelpick__confirm-body">
            {run!.machines > 1 ? t("model.switch.cluster", { n: run!.machines - 1 }) : t("model.switch.solo")}
          </p>
          <div className="modelpick__actions">
            <button className="btn-secondary" onClick={() => setPending(null)}>
              {t("model.switch.cancel")}
            </button>
            <button className="btn-primary" onClick={confirm}>
              {t("model.switch.go")}
            </button>
          </div>
        </div>
      ) : (
        <>
          {/* The label IS the signal that the order below came from usage data.
              Absent = the static manifest order (see the fetch above). */}
          {usage ? <p className="modelpick__ranked">{t("model.rank.byUsage")}</p> : null}
          <div className="modelpick__list">
            {ordered.map((m) => {
              const tokens = usage?.get(m.id);
              const mode = fitOf(m.id, m.id === sel.modelId ? sel.quant : defaultQuant(m.id));
              return (
                <button
                  key={m.id}
                  className={`modelpick__item${m.id === sel.modelId ? " is-on" : ""}`}
                  onClick={() => setSel({ modelId: m.id, quant: defaultQuant(m.id) })}
                >
                  <span className="modelpick__name">{m.label}</span>
                  {/* Whether a model can be pooled changes what the Cluster screen
                      will let you do next, and it is not guessable from the size. */}
                  <span className="modelpick__deploy">
                    {t(isSingleNode(m.id) ? "settings.model.singleNode" : "settings.model.cluster")}
                  </span>
                  <span className="modelpick__meta">
                    <span className="modelpick__params">{m.params}</span>
                    {/* Can these machines actually run it — the advisor's verdict.
                        Since 2026-08-21 this chip is the ONLY place it appears:
                        the capability table was cut to four columns and the
                        "Can run" one went with the rest. */}
                    {mode ? <span className={`modelpick__fit ${fitClass(mode)}`}>{t(fitKey(mode))}</span> : null}
                    {/* Usage only when there IS usage: a "0 tokens" chip reads as
                        "nobody runs this", which is not what missing data means. */}
                    {tokens ? (
                      <span className="modelpick__usage">{t("model.rank.tokens", { n: shortCount(tokens) })}</span>
                    ) : null}
                  </span>
                </button>
              );
            })}
          </div>
          {/* A model you want that is not listed: the answer is a GitHub issue,
              not a file box — see model.request.hint. The sentence used to say
              "request it in a GitHub issue" with nothing to click (A-P1-4). */}
          <p className="modelpick__open-hint">
            {t("model.request.hint")}{" "}
            <button className="linkbtn" onClick={() => void openExternal(ISSUES_URL)}>
              {t("model.request.link")} ↗
            </button>
          </p>
          {/* Precision belongs to the selected model, so it stays a separate row
              rather than multiplying the list by five. */}
          {hasQuantChoice(sel.modelId) ? (
            <div className="modelpick__quant">
              <span className="modelpick__quant-label">{t("settings.precision")}</span>
              <select
                className="select"
                value={sel.quant || defaultQuant(sel.modelId)}
                onChange={(e) => setSel((p) => ({ ...p, quant: e.target.value }))}
              >
                {quantOptions(sel.modelId).map((v) => (
                  <option key={v.quant} value={v.quant}>
                    {v.quant} · {fmtBytes(v.layer_weight_bytes + v.shared_weight_bytes)}
                  </option>
                ))}
              </select>
            </div>
          ) : null}
          <div className="modelpick__actions">
            {/* A-P2-6: "Keep current" belongs to the CONFIRM step above, where
                the alternative really is keeping the running model. Here
                nothing has been changed yet, so the button that closes the
                list is just Cancel — "Keep current" read as a second choice
                and made people look for the one that says "don't apply". */}
            <button className="btn-secondary" onClick={props.onClose}>
              {t("weights.cancel")}
            </button>
            <button className="btn-primary" onClick={() => choose(sel.modelId, sel.quant)}>
              {t("model.pick.apply")}
            </button>
          </div>
        </>
      )}
    </div>
  );
}
