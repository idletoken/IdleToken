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
import {
  AVAILABLE_MODELS,
  MODEL_BRANDS,
  defaultQuant,
  quantOptions,
  shortModelLabel,
} from "./models";
import { fmtBytes, fmtQuant } from "./format";
import { useDialog } from "./useDialog";
import { loadLocalCapability, type CapabilityMode, type CapabilityRow } from "./Capability";
import { fetchLeaderboard } from "./platform";

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

/** The local advisor's three states. These labels deliberately say "locally":
 *  a cluster may still run a model that this one machine cannot. "unavailable"
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

type Pick = { modelId: string; quant: string };

/** The family a model belongs to (the card it is listed under). */
function brandIdOf(modelId: string): string {
  return MODEL_BRANDS.find((b) => b.models.some((m) => m.id === modelId))?.id ?? "";
}

// The last APPLIED pick per family, kept across launches. Opening a family
// lands on it, so coming back to "Qwen3.5" a week later starts from the size
// you actually used rather than from the smallest one.
const LAST_PICK_KEY = "idletoken.picker.lastByFamily";
function loadLastPicks(): Record<string, Pick> {
  try {
    const raw = localStorage.getItem(LAST_PICK_KEY);
    const v = raw ? (JSON.parse(raw) as unknown) : null;
    return v && typeof v === "object" ? (v as Record<string, Pick>) : {};
  } catch {
    return {};
  }
}
function saveLastPick(pick: Pick): void {
  const brand = brandIdOf(pick.modelId);
  if (!brand) return;
  try {
    localStorage.setItem(LAST_PICK_KEY, JSON.stringify({ ...loadLastPicks(), [brand]: pick }));
  } catch {
    /* storage unavailable — the pick still applies, only the memory of it is lost */
  }
}

export default function ModelPicker(props: {
  /** The local setting: what the next start would use. */
  modelId: string;
  quant: string;
  running: RunningModel | null;
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

  // The three-state LOCAL fit verdict per model. It always comes from this
  // machine's advisor, even while a cluster is online: otherwise the same chip
  // would silently change from "this machine" to "whole cluster" depending on
  // connection state. A second derivation in TypeScript remains forbidden.
  const [fit, setFit] = useState<Map<string, CapabilityRow[]>>(new Map());
  useEffect(() => {
    let alive = true;
    loadLocalCapability()
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
  }, []);
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

  /** The one place a pick leaves the picker: remember it for its family, hand
   *  it to the caller, close. */
  const commit = (pick: Pick) => {
    saveLastPick(pick);
    props.onPick(pick.modelId, pick.quant);
    props.onClose();
  };

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
      commit({ modelId, quant });
      return;
    }
    setPending({ modelId, quant });
  };

  const confirm = () => {
    if (!pending) return;
    commit(pending);
  };

  const curQuant = cur.quant || defaultQuant(cur.modelId);

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
  const [sel, setSel] = useState<Pick>({ modelId: cur.modelId, quant: curQuant });
  const initialBrand = MODEL_BRANDS.find((b) => b.models.some((m) => m.id === cur.modelId));
  const [brandId, setBrandId] = useState(initialBrand?.id ?? MODEL_BRANDS[0]?.id ?? "");
  const [level, setLevel] = useState<"families" | "models">("families");
  // Drafts per family for THIS opening of the picker: pick 27B under Qwen3.5,
  // look at GLM, come back — 27B is still the draft.
  const [drafts, setDrafts] = useState<Record<string, Pick>>({});
  const [lastPicks] = useState(loadLastPicks);
  const pickDraft = (pick: Pick) => {
    setSel(pick);
    const brand = brandIdOf(pick.modelId);
    if (brand) setDrafts((d) => ({ ...d, [brand]: pick }));
  };
  const quantValid = (modelId: string, quant: string) =>
    quantOptions(modelId).some((v) => v.quant === quant);

  // Usage rank (or manifest order) is the base sequence; runnable models then
  // come first. "Unknown" sits between runnable and won't-run so a row does
  // not jump to the bottom merely because the advisor is still loading.
  const ranked = usage
    ? [...AVAILABLE_MODELS].sort((a, b) => (usage.get(b.id) ?? -1) - (usage.get(a.id) ?? -1))
    : AVAILABLE_MODELS;
  const runRank = (m: (typeof AVAILABLE_MODELS)[number]): number => {
    const mode = fitOf(m.id, m.id === sel.modelId ? sel.quant : defaultQuant(m.id));
    if (mode === "gpu_only") return 0;
    if (mode === "hybrid") return 1;
    return mode ? 3 : 2;
  };
  // Array.prototype.sort is stable: equal ranks keep the usage/manifest order.
  const ordered = [...ranked].sort((a, b) => runRank(a) - runRank(b));
  const activeBrand = MODEL_BRANDS.find((b) => b.id === brandId) ?? MODEL_BRANDS[0];
  const modelsOf = (brand: (typeof MODEL_BRANDS)[number]) =>
    ordered.filter((m) => brand.models.some((candidate) => candidate.id === m.id));
  const familyModels = activeBrand ? modelsOf(activeBrand) : [];
  const selectedQuantOptions = quantOptions(sel.modelId);

  // Opening a family always lands the draft INSIDE it (2026-09-07, owner's
  // call). Before, the draft stayed on the current model — 0.8B under Qwen3.5
  // — while you were looking at another family, so the precision dropdown
  // belonged to a row that was not even on screen, and people read the list
  // as if the size they were looking at were the chosen one. In order: the
  // draft if it is already here, this opening's draft for the family, the
  // current model, the last pick applied under this family, the first row.
  const enterFamily = (brand: (typeof MODEL_BRANDS)[number]) => {
    const models = modelsOf(brand);
    const here = (id: string) => models.some((m) => m.id === id);
    let next: Pick | null = null;
    const draft = drafts[brand.id];
    const last = lastPicks[brand.id];
    if (here(sel.modelId)) next = sel;
    else if (draft && here(draft.modelId)) next = draft;
    else if (here(cur.modelId)) next = { modelId: cur.modelId, quant: curQuant };
    else if (last && here(last.modelId)) {
      next = {
        modelId: last.modelId,
        quant: quantValid(last.modelId, last.quant) ? last.quant : defaultQuant(last.modelId),
      };
    } else if (models[0]) next = { modelId: models[0].id, quant: defaultQuant(models[0].id) };
    if (next) pickDraft(next);
    setBrandId(brand.id);
    setLevel("models");
  };

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
          <div className="modelpick__nav">
            {level === "models" ? (
              <button
                type="button"
                className="modelpick__back"
                onClick={() => setLevel("families")}
                aria-label={t("model.pick.back")}
              >
                <span aria-hidden="true">←</span>
                {t("model.pick.back")}
              </button>
            ) : null}
            <span className="modelpick__nav-title">
              {level === "families" ? t("model.pick.family") : activeBrand?.label}
            </span>
          </div>
          <div className="modelpick__list">
            {level === "families" ? MODEL_BRANDS.map((brand) => {
              // The family row names the CURRENT model, not the draft: since
              // opening a family moves the draft into it, following the draft
              // here would make merely browsing GLM relabel the GLM row.
              const selected = brand.models.find((m) => m.id === cur.modelId);
              return (
                <button
                  type="button"
                  key={brand.id}
                  className={`modelpick__item modelpick__family${selected ? " is-on" : ""}`}
                  onClick={() => enterFamily(brand)}
                >
                  <span className="modelpick__name">{brand.label}</span>
                  <span className="modelpick__family-meta">
                    {/* Every family row says how many models it holds; the one
                        holding the current pick adds which (2026-09-06: showing
                        only the pick made that row look like it had one model). */}
                    {selected
                      ? t("model.pick.familyCurrent", {
                          n: brand.models.length,
                          model: shortModelLabel(selected.label, brand.label),
                        })
                      : t("model.pick.modelCount", { n: brand.models.length })}
                  </span>
                  <span className="modelpick__chevron" aria-hidden="true">›</span>
                </button>
              );
            }) : familyModels.map((m) => {
              const mode = fitOf(m.id, m.id === sel.modelId ? sel.quant : defaultQuant(m.id));
              return (
                <button
                  type="button"
                  key={m.id}
                  className={`modelpick__item${m.id === sel.modelId ? " is-on" : ""}`}
                  aria-pressed={m.id === sel.modelId}
                  onClick={() => pickDraft({ modelId: m.id, quant: defaultQuant(m.id) })}
                >
                  <span className="modelpick__name">
                    {activeBrand ? shortModelLabel(m.label, activeBrand.label) : m.label}
                  </span>
                  <span className="modelpick__meta">
                    {/* Parameter counts cut 2026-08-26 (owner's call): "1T ·
                        32B active" does not help the pick; the name and the
                        fit chip do. */}
                    {/* Can these machines actually run it — the advisor's verdict.
                        Since 2026-08-21 this chip is the ONLY place it appears:
                        the capability table was cut to four columns and the
                        "Can run" one went with the rest. */}
                    {mode ? <span className={`modelpick__fit ${fitClass(mode)}`}>{t(fitKey(mode))}</span> : null}
                    {/* The 7-day token chip was cut 2026-08-26 (owner's call):
                        platform usage numbers do not help someone pick a model
                        to RUN HERE. Ranking still uses the usage ordering. */}
                  </span>
                </button>
              );
            })}
          </div>
          {/* Precision belongs to the selected model, so it stays a separate row
              rather than multiplying the list by five. */}
          {level === "models" && selectedQuantOptions.length > 0 ? (
            <div className="modelpick__quant">
              <span className="modelpick__quant-label">{t("settings.precision")}</span>
              <select
                className="select"
                value={sel.quant || defaultQuant(sel.modelId)}
                onChange={(e) => pickDraft({ modelId: sel.modelId, quant: e.target.value })}
              >
                {selectedQuantOptions.map((v) => (
                  <option key={v.quant} value={v.quant}>
                    {fmtQuant(v.quant)} · {fmtBytes(v.layer_weight_bytes + v.shared_weight_bytes)}
                  </option>
                ))}
              </select>
            </div>
          ) : null}
          <div className={`modelpick__actions${level === "families" ? " modelpick__actions--families" : ""}`}>
            {/* A-P2-6: "Keep current" belongs to the CONFIRM step above, where
                the alternative really is keeping the running model. Here
                nothing has been changed yet, so the button that closes the
                list is just Cancel — "Keep current" read as a second choice
                and made people look for the one that says "don't apply". */}
            <button type="button" className="btn-secondary" onClick={props.onClose}>
              {t("weights.cancel")}
            </button>
            {level === "models" ? (
              <button type="button" className="btn-primary" onClick={() => choose(sel.modelId, sel.quant)}>
                {t("model.pick.apply")}
              </button>
            ) : null}
          </div>
        </>
      )}
    </div>
  );
}
