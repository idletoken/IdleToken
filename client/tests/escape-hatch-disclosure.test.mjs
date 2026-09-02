import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const mainSource = readFileSync(
  new URL("../src-tauri/src/main.rs", import.meta.url),
  "utf8",
);
const resourceSource = readFileSync(
  new URL("../../src/common/resource.c", import.meta.url),
  "utf8",
);

function blockBetween(source, startNeedle, endNeedle) {
  const start = source.indexOf(startNeedle);
  if (start < 0) return "";
  const end = source.indexOf(endNeedle, start + startNeedle.length);
  return end < 0 ? "" : source.slice(start, end);
}

function uiDisclosureProblems(source) {
  const problems = [];
  const mainStart = source.indexOf("fn main() {");
  const startupCall = source.indexOf(
    "let _ = configured_ui_test_directives();",
    mainStart,
  );
  const builder = source.indexOf("tauri::Builder::default()", mainStart);
  if (mainStart < 0 || startupCall < 0 || builder < 0 || startupCall > builder) {
    problems.push("startup disclosure is not initialized before Tauri");
  } else if (
    source.slice(mainStart + "fn main() {".length, startupCall).trim() !== ""
  ) {
    problems.push("startup disclosure is not the first main statement");
  }

  const initializer = blockBetween(
    source,
    "fn initialize_ui_test_directives(",
    "fn configured_ui_test_directives()",
  );
  const emit = initializer.indexOf("disclose(UI_TEST_DISCLOSURE);");
  const returnDirectives = initializer.lastIndexOf("directives\n}");
  if (
    !initializer.includes("if !directives.is_empty()") ||
    emit < 0 ||
    returnDirectives < 0 ||
    emit > returnDirectives
  ) {
    problems.push("non-empty directives are not disclosed before return");
  }
  if (/disclose\s*\(\s*(raw|directives)/.test(initializer)) {
    problems.push("the disclosure can include directive contents");
  }

  const configured = blockBetween(
    source,
    "fn configured_ui_test_directives()",
    "/// UI-test channel",
  );
  if (
    !configured.includes("UI_TEST_DIRECTIVES") ||
    !configured.includes(".get_or_init(") ||
    !configured.includes('std::env::var("IDLETOKEN_UI_TEST")') ||
    !configured.includes('eprintln!("{line}")')
  ) {
    problems.push("the process-wide directive snapshot is not single and loud");
  }

  const command = blockBetween(
    source,
    "fn ui_test_directives()",
    "#[cfg(test)]\nmod ui_test_disclosure_tests",
  );
  if (!command.includes("configured_ui_test_directives().to_vec()")) {
    problems.push("the IPC command can bypass the disclosed directive snapshot");
  }
  return problems;
}

function windowsNvmlProblems(source) {
  const problems = [];
  const windows = blockBetween(
    source,
    "#if defined(_WIN32)",
    "#elif defined(__APPLE__)",
  );
  const forcedStart = windows.indexOf(
    'if (getenv("IDLETOKEN_FORCE_NO_NVML")) {',
  );
  const load = windows.indexOf('LoadLibraryA("nvml.dll")');
  const forcedBlock = blockBetween(
    windows,
    'if (getenv("IDLETOKEN_FORCE_NO_NVML")) {',
    "    HMODULE h = LoadLibraryA",
  );
  if (forcedStart < 0 || load < 0 || forcedStart > load) {
    problems.push("the forced branch does not short-circuit before NVML loading");
  }
  if (
    !forcedBlock.includes("fprintf(stderr") ||
    !forcedBlock.includes("forced by IDLETOKEN_FORCE_NO_NVML") ||
    forcedBlock.indexOf("fprintf(stderr") > forcedBlock.indexOf("return NULL;")
  ) {
    problems.push("the forced branch does not print its explicit diagnostic first");
  }

  const genuineMissing = blockBetween(
    windows,
    "if (!nv) {",
    "fn_nvmlInit",
  );
  if (
    !genuineMissing.includes("nvml.dll not found") ||
    genuineMissing.includes("forced") ||
    forcedBlock.includes("nvml.dll not found")
  ) {
    problems.push("forced and genuine missing-driver diagnostics are not distinct");
  }
  return problems;
}

test("current sources enforce both loud escape-hatch contracts", () => {
  assert.deepEqual(uiDisclosureProblems(mainSource), []);
  assert.deepEqual(windowsNvmlProblems(resourceSource), []);
});

test("UI inverse mutations turn the source contract red", () => {
  const noStartup = mainSource.replace(
    "    let _ = configured_ui_test_directives();\n",
    "",
  );
  assert.notEqual(noStartup, mainSource, "control mutation must change the source");
  assert.match(uiDisclosureProblems(noStartup).join("\n"), /before Tauri/);

  const silent = mainSource.replace(
    "        disclose(UI_TEST_DISCLOSURE);",
    "        // inverse control: disclosure removed",
  );
  assert.notEqual(silent, mainSource, "control mutation must change the source");
  assert.match(uiDisclosureProblems(silent).join("\n"), /before return/);

  const leaksContents = mainSource.replace(
    "        disclose(UI_TEST_DISCLOSURE);",
    "        disclose(raw.unwrap_or_default());",
  );
  assert.notEqual(leaksContents, mainSource, "control mutation must change the source");
  assert.match(
    uiDisclosureProblems(leaksContents).join("\n"),
    /before return|include directive contents/,
  );
});

test("NVML inverse mutations turn the source contract red", () => {
  const silent = resourceSource.replace(
    '        fprintf(stderr, "idletoken-probe: *** TEST OVERRIDE ACTIVE *** "',
    '        /* inverse control: forced diagnostic removed */ (void)""',
  );
  assert.notEqual(silent, resourceSource, "control mutation must change the source");
  assert.match(windowsNvmlProblems(silent).join("\n"), /explicit diagnostic/);

  const indistinguishable = resourceSource.replace(
    "nvml.dll not found",
    "NVML unavailable (forced)",
  );
  assert.notEqual(
    indistinguishable,
    resourceSource,
    "control mutation must change the source",
  );
  assert.match(
    windowsNvmlProblems(indistinguishable).join("\n"),
    /not distinct/,
  );
});
