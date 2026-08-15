// "What is taking up my disk, and let me get it back."
//
// A scan of the model folder rather than a lookup of the models we know about:
// what actually accumulates there is the precision you tried once, the model you
// switched away from, and the half of a download you cancelled — and a list
// built from the manifests can only show what it thought to ask for. Anything
// found that no manifest claims is still listed, by file name; it is on the
// disk either way, and hiding it would defeat the point of the screen.
import { useCallback, useEffect, useState } from "react";
import { useI18n } from "./i18n";
import { fmtBytes } from "./format";
import { describeGguf } from "./models";
import { defaultModelDir, deleteWeights, listWeights, type StoredWeights } from "./weights";

export default function StoredModels(props: {
  /** settings.modelDir; empty = the platform default folder. */
  modelDir: string;
  /** Tell the app to re-check the selected model's weights after a deletion —
   *  otherwise the row above still says "ready" for a file that is gone. */
  onChanged?: () => void;
}) {
  const { t } = useI18n();
  const [dir, setDir] = useState("");
  const [files, setFiles] = useState<StoredWeights[] | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const d = props.modelDir || (await defaultModelDir());
      setDir(d);
      setFiles(await listWeights(d));
      setError(null);
    } catch (e) {
      // Outside Tauri (browser dev build) there is no folder to read. Say so
      // once instead of rendering an empty list that claims the disk is clean.
      setFiles([]);
      setError(String(e));
    }
  }, [props.modelDir]);

  useEffect(() => {
    void load();
  }, [load]);

  const remove = async (file: string) => {
    setBusy(file);
    setConfirming(null);
    try {
      await deleteWeights(dir, file);
      setError(null);
      await load();
      props.onChanged?.();
    } catch (e) {
      // The common one is Windows refusing to delete a file the running engine
      // still has open — the message says which file, so it is actionable.
      setError(String(e));
    } finally {
      setBusy(null);
    }
  };

  if (files === null) return <p className="setting-hint">{t("store.loading")}</p>;

  const total = files.reduce((n, f) => n + f.bytes, 0);

  return (
    <div className="stored">
      <div className="stored__head">
        <span className="stored__dir" title={dir}>{dir || "—"}</span>
        {files.length > 0 ? (
          <span className="stored__total">{t("store.total", { size: fmtBytes(total) })}</span>
        ) : null}
      </div>

      {files.length === 0 ? (
        <p className="setting-hint">{error ? t("store.unreadable") : t("store.empty")}</p>
      ) : (
        <ul className="stored__list">
          {files.map((f) => {
            const d = describeGguf(f.file);
            return (
              <li key={f.file + (f.partial ? ".part" : "")} className="stored__item">
                <span className="stored__name" title={f.file}>
                  {d ? d.label : f.file}
                  {d?.quant ? <span className="stored__quant">{d.quant}</span> : null}
                  {/* An unfinished download is worth its own word: it is the one
                      row where deleting throws away progress that would
                      otherwise be resumed rather than re-downloaded. */}
                  {f.partial ? <span className="stored__partial">{t("store.partial")}</span> : null}
                </span>
                <span className="stored__size">{fmtBytes(f.bytes)}</span>
                {confirming === f.file ? (
                  <span className="stored__confirm">
                    <span className="stored__ask">
                      {f.partial ? t("store.askPartial") : t("store.ask")}
                    </span>
                    <button className="linkbtn" onClick={() => setConfirming(null)}>
                      {t("store.keep")}
                    </button>
                    <button className="linkbtn stored__go" onClick={() => void remove(f.file)}>
                      {t("store.confirm")}
                    </button>
                  </span>
                ) : (
                  <button
                    className="linkbtn"
                    disabled={busy === f.file}
                    onClick={() => setConfirming(f.file)}
                  >
                    {busy === f.file ? t("store.deleting") : t("store.delete")}
                  </button>
                )}
              </li>
            );
          })}
        </ul>
      )}

      {/* Delete failures (the common one: Windows refusing to remove a file
          the running engine still holds). Localized headline; the OS/engine
          sentence stays verbatim as the detail — it names the file. */}
      {error && files.length > 0 ? (
        <p className="setting-hint stored__err">{t("store.deleteFailed", { detail: error })}</p>
      ) : null}
    </div>
  );
}
