import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { hydrateSecrets } from "./secrets";
import "./fonts.css";
import "./styles.css";

// The platform session no longer lives in localStorage (audit A-P2-2), so it
// has to be read from the host process before anything renders. It must be
// BEFORE, not in an effect: `AuthProvider.currentSession()` is synchronous and
// App reads it in its initial state, so a later hydration would leave a signed-
// in user looking signed out until something else happened to re-render.
//
// One round trip over IPC, no network. A failure inside hydrateSecrets is
// already swallowed there (it falls back to an empty store, i.e. signed out),
// so the app starts either way rather than showing a blank window.
void hydrateSecrets().finally(() => {
  ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
});
