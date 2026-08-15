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
import { useEffect, useState } from "react";
import { useI18n } from "./i18n";
import { AVAILABLE_MODELS, isLocalGguf, defaultQuant, hasQuantChoice, quantOptions, isSingleNode } from "./models";
import type { CustomModelSource } from "./weights";
import { fmtBytes } from "./format";
import { useDialog } from "./useDialog";
import { inTauri } from "./platform";

/** Parse the HF inputs into a repo + exact .gguf file name. The first box
 *  accepts either "owner/name" or a full huggingface.co file link (blob/ or
 *  resolve/ URL) — pasting the browser address bar is the common case, and
 *  asking someone to dissect it by hand is error entry by design. */
export function parseHfInput(repoOrUrl: string, fileName: string): { repo: string; file: string } | null {
  let repo = repoOrUrl.trim();
  let file = fileName.trim();
  const m = repo.match(/huggingface\.co\/([^/\s]+\/[^/\s]+)\/(?:blob|resolve)\/[^/\s]+\/([^?\s]+\.gguf)/i);
  if (m) {
    repo = m[1];
    if (!file) file = decodeURIComponent(m[2]);
  }
  if (!/^[\w][\w.-]*\/[\w][\w.-]*$/.test(repo)) return null;
  if (!/\.gguf$/i.test(file) || file.includes("/")) return null;
  return { repo, file };
}

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

export default function ModelPicker(props: {
  /** The local setting: what the next start would use. */
  modelId: string;
  quant: string;
  running: RunningModel | null;
  /** Apply the pick. The caller saves it and performs any restart. */
  onPick: (modelId: string, quant: string) => void;
  /** Apply an OPEN pick — a user-supplied GGUF (local file / HF repo+file).
   *  The caller saves it under LOCAL_GGUF_ID and restarts through the local
   *  llama.cpp engine. Absent = this surface cannot switch to open models. */
  onPickCustom?: (c: CustomModelSource) => void;
  /** Display name of the current open selection (file name), when the setting
   *  points at one. */
  customName?: string;
  onClose: () => void;
}) {
  const { t } = useI18n();
  const ref = useDialog(props.onClose);
  // A pick that is waiting for confirmation because it would restart something.
  const [pending, setPending] = useState<
    { kind: "curated"; modelId: string; quant: string } | { kind: "open"; c: CustomModelSource; label: string } | null
  >(null);
  const [hfOpen, setHfOpen] = useState(false);
  const [hfRepo, setHfRepo] = useState("");
  const [hfFile, setHfFile] = useState("");
  const [hfErr, setHfErr] = useState(false);

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
    setPending({ kind: "curated", modelId, quant });
  };

  /** An open pick goes through the same restart gate as a curated one: with
   *  something running it is a rebuild, and the picker says so first. */
  const chooseCustom = (c: CustomModelSource, label: string) => {
    if (!props.onPickCustom) return;
    if (run && run.modelId) {
      setPending({ kind: "open", c, label });
      return;
    }
    props.onPickCustom(c);
    props.onClose();
  };

  const pickLocalFile = async () => {
    if (!inTauri()) return; // button carries the explanation
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const picked = await open({
        multiple: false,
        directory: false,
        filters: [{ name: "GGUF", extensions: ["gguf"] }],
      });
      const path = typeof picked === "string" ? picked : null;
      if (!path) return; // cancelled
      const cut = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
      chooseCustom({ source: "file", path, repo: "", file: "" }, cut >= 0 ? path.slice(cut + 1) : path);
    } catch (e) {
      console.error("gguf file dialog:", e);
    }
  };

  const useHf = () => {
    const parsed = parseHfInput(hfRepo, hfFile);
    if (!parsed) {
      setHfErr(true);
      return;
    }
    setHfErr(false);
    chooseCustom({ source: "hf", path: "", repo: parsed.repo, file: parsed.file }, parsed.file);
  };

  const confirm = () => {
    if (!pending) return;
    if (pending.kind === "curated") props.onPick(pending.modelId, pending.quant);
    else props.onPickCustom?.(pending.c);
    props.onClose();
  };

  const curQuant = cur.quant || defaultQuant(cur.modelId);

  return (
    <div className="modelpick" ref={ref} role="dialog" aria-label={t("model.pick.title")}>
      {pending ? (
        <div className="modelpick__confirm">
          <p className="modelpick__confirm-title">
            {t("model.switch.title", {
              model:
                pending.kind === "curated"
                  ? AVAILABLE_MODELS.find((m) => m.id === pending.modelId)?.label ?? pending.modelId
                  : pending.label,
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
          <div className="modelpick__list">
            {AVAILABLE_MODELS.map((m) => (
              <button
                key={m.id}
                className={`modelpick__item${m.id === cur.modelId ? " is-on" : ""}`}
                onClick={() => choose(m.id, defaultQuant(m.id))}
              >
                <span className="modelpick__name">{m.label}</span>
                <span className="modelpick__params">{m.params}</span>
                {/* Whether a model can be pooled changes what the Cluster screen
                    will let you do next, and it is not guessable from the size. */}
                <span className="modelpick__deploy">
                  {t(isSingleNode(m.id) ? "settings.model.singleNode" : "settings.model.cluster")}
                </span>
              </button>
            ))}
          </div>
          {/* Open intake (v2 WS-D1): any GGUF llama.cpp can run, below the
              curated recommendations. The engine reads the manifest from the
              file and rules on fit at start — the picker only says where the
              file lives. */}
          {props.onPickCustom ? (
            <div className="modelpick__open">
              <div className="modelpick__open-head">
                <span>{t("model.open.section")}</span>
                {isLocalGguf(cur.modelId) && props.customName ? (
                  <span className="modelpick__open-cur is-on" title={props.customName}>
                    {props.customName}
                  </span>
                ) : null}
              </div>
              <div className="modelpick__open-actions">
                <button
                  className="btn-secondary"
                  disabled={!inTauri()}
                  title={inTauri() ? undefined : t("model.open.pickBrowser")}
                  onClick={() => void pickLocalFile()}
                >
                  {t("model.open.pickFile")}
                </button>
                <button className="linkbtn" onClick={() => setHfOpen((v) => !v)} aria-expanded={hfOpen}>
                  {t("model.open.hfToggle")}
                </button>
              </div>
              {hfOpen ? (
                <div className="modelpick__hf">
                  <input
                    className="field__input"
                    value={hfRepo}
                    placeholder={t("model.open.hfRepo")}
                    onChange={(e) => setHfRepo(e.target.value)}
                  />
                  <input
                    className="field__input"
                    value={hfFile}
                    placeholder={t("model.open.hfFile")}
                    onChange={(e) => setHfFile(e.target.value)}
                  />
                  <button className="btn-secondary" onClick={useHf}>
                    {t("model.open.hfUse")}
                  </button>
                  {hfErr ? <p className="modelpick__hf-err">{t("model.open.hfBad")}</p> : null}
                </div>
              ) : null}
              <p className="modelpick__open-hint">{t("model.open.hint")}</p>
            </div>
          ) : null}
          {/* Precision belongs to the selected model, so it stays a separate row
              rather than multiplying the list by five. (An open GGUF has no
              menu — its precision is whatever the file is.) */}
          {!isLocalGguf(cur.modelId) && hasQuantChoice(cur.modelId) ? (
            <div className="modelpick__quant">
              <span className="modelpick__quant-label">{t("settings.precision")}</span>
              <select
                className="select"
                value={curQuant}
                onChange={(e) => choose(cur.modelId, e.target.value)}
              >
                {quantOptions(cur.modelId).map((v) => (
                  <option key={v.quant} value={v.quant}>
                    {v.quant} · {fmtBytes(v.layer_weight_bytes + v.shared_weight_bytes)}
                  </option>
                ))}
              </select>
            </div>
          ) : null}
        </>
      )}
    </div>
  );
}
