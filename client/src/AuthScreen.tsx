import { useState } from "react";
import { useI18n, type StringKey } from "./i18n";
import { AuthError, getAuthProvider, openExternal, portalRegisterUrl, resendVerification, type ResendResult, type Session } from "./auth";
import { portalSignInUrl } from "./links";
import { useDialog } from "./useDialog";

type Mode = "signIn" | "signUp";

export default function AuthScreen(props: {
  onAuthed: (s: Session) => void;
  onClose: () => void;
}) {
  const dialogRef = useDialog(props.onClose);
  const { t } = useI18n();
  // Registration lives on the website (see portalRegisterUrl). The local signup
  // form survives ONLY in builds with no platform configured, where there is no
  // website and the local provider is the sole way to get an identity.
  const registerUrl = portalRegisterUrl();
  // A-P1-5: a forgotten password had no exit at all — the reset link arrives by
  // email and lands on the website, so the client cannot own the flow, but it
  // can stop being a dead end. Absent in an offline build, where the local
  // provider keeps the only account and there is no website to send anyone to.
  const resetUrl = portalSignInUrl();
  const [mode, setMode] = useState<Mode>("signIn");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<StringKey | null>(null);
  const [resent, setResent] = useState<ResendResult | null>(null);

  const resend = async () => {
    setResent(await resendVerification(email));
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(null);
    setResent(null);
    try {
      const auth = getAuthProvider();
      const s = mode === "signUp" ? await auth.signUp(email, password) : await auth.signIn(email, password);
      props.onAuthed(s);
    } catch (err) {
      setError(err instanceof AuthError ? (err.code as StringKey) : "auth.err.server");
      setBusy(false);
    }
  };

  // Which backend the selector will use (depends on the platformUrl setting).
  const providerKind = getAuthProvider().kind;

  return (
    <div className="modal-scrim" onClick={props.onClose}>
      <div
        className="modal modal--auth"
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label={t("auth.title")}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal__head">
          <div>
            <h2>{t("auth.title")}</h2>
            <p className="auth-subtitle">{t("auth.subtitle")}</p>
          </div>
          <button className="iconbtn" onClick={props.onClose} aria-label={t("a11y.close")}>
            ✕
          </button>
        </div>

        {registerUrl ? null : (
          <div className="auth-tabs" role="tablist">
            <button
              role="tab"
              aria-selected={mode === "signIn"}
              className={`auth-tab${mode === "signIn" ? " is-on" : ""}`}
              onClick={() => {
                setMode("signIn");
                setError(null);
              }}
            >
              {t("auth.tabSignIn")}
            </button>
            <button
              role="tab"
              aria-selected={mode === "signUp"}
              className={`auth-tab${mode === "signUp" ? " is-on" : ""}`}
              onClick={() => {
                setMode("signUp");
                setError(null);
              }}
            >
              {t("auth.tabSignUp")}
            </button>
          </div>
        )}

        <form onSubmit={submit}>
          <label className="field">
            <span className="field__k">{t("auth.email")}</span>
            <input
              className="field__input"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
            />
          </label>
          <label className="field">
            <span className="field__k">{t("auth.password")}</span>
            <input
              className="field__input"
              type="password"
              autoComplete={mode === "signUp" ? "new-password" : "current-password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
            />
            {mode === "signUp" ? <span className="field__hint">{t("auth.passwordHint")}</span> : null}
            {mode === "signIn" && resetUrl ? (
              <button type="button" className="linkbtn field__hint" onClick={() => openExternal(resetUrl)}>
                {t("auth.forgotPassword")} ↗
              </button>
            ) : null}
          </label>

          {error ? (
            <div className="auth-error" role="alert">
              {t(error)}
              {/* Being told "verify your email" with no way to get another one
                  is a dead end — the resend endpoint needs no session. */}
              {error === "auth.err.unverified" ? (
                <>
                  {" "}
                  {resent === null ? (
                    <button type="button" className="linkbtn" onClick={resend}>
                      {t("auth.resendVerify")}
                    </button>
                  ) : (
                    <span className="auth-error__sent">
                      {resent === "sent"
                        ? t("auth.verifySent")
                        : resent === "rate-limited"
                          ? t("auth.verifyTooOften")
                          : t("auth.verifyFailed")}
                    </span>
                  )}
                </>
              ) : null}
            </div>
          ) : null}

          <button className="btn-primary btn-block" type="submit" disabled={busy}>
            {busy ? t("auth.working") : mode === "signUp" ? t("auth.submitSignUp") : t("auth.submitSignIn")}
          </button>
        </form>

        {registerUrl ? (
          <p className="auth-signup">
            {t("auth.noAccount")}{" "}
            <button className="linkbtn" onClick={() => openExternal(registerUrl)}>
              {t("auth.registerOnWeb")} ↗
            </button>
          </p>
        ) : null}

        <p className="auth-note">{providerKind === "cloud" ? t("auth.cloudNote") : t("auth.localNote")}</p>
      </div>
    </div>
  );
}
