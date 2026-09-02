import { createHash } from "node:crypto";
import { readdir, readFile, stat, writeFile } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const clientRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const manifestPath = join(clientRoot, "dist", "frontend-provenance.json");
const mode = "release";

const sourceDirs = [
  "src",
  "public",
  "../packages/shared-ui/src",
];
const sourceFiles = [
  "index.html",
  "package.json",
  "pnpm-lock.yaml",
  "tsconfig.json",
  "tsconfig.node.json",
  "vite.config.ts",
  ".env.release",
  "../packages/shared-ui/package.json",
];

const slash = (p) => p.replaceAll("\\", "/");
const rel = (p) => slash(relative(clientRoot, p));

async function existingFile(path) {
  try {
    return (await stat(path)).isFile();
  } catch {
    return false;
  }
}

async function walk(dir, out) {
  let entries;
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch (error) {
    if (error?.code === "ENOENT") return;
    throw error;
  }
  entries.sort((a, b) => a.name.localeCompare(b.name));
  for (const entry of entries) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) await walk(path, out);
    else if (entry.isFile()) out.push(path);
  }
}

async function currentFiles() {
  const files = [];
  for (const dir of sourceDirs) await walk(resolve(clientRoot, dir), files);
  for (const file of sourceFiles) {
    const path = resolve(clientRoot, file);
    if (await existingFile(path)) files.push(path);
  }
  return [...new Set(files.map((p) => resolve(p)))].sort((a, b) => rel(a).localeCompare(rel(b)));
}

async function digest(path) {
  return createHash("sha256").update(await readFile(path)).digest("hex");
}

async function snapshot() {
  const packageJson = JSON.parse(await readFile(join(clientRoot, "package.json"), "utf8"));
  const files = {};
  for (const path of await currentFiles()) files[rel(path)] = await digest(path);
  return { schema: 1, mode, packageVersion: packageJson.version, files };
}

const current = await snapshot();
if (process.argv.includes("--verify")) {
  const saved = JSON.parse(await readFile(manifestPath, "utf8"));
  if (JSON.stringify(saved) !== JSON.stringify(current)) {
    throw new Error("client/dist is stale: rebuild it with `pnpm build:release`");
  }
  console.log(`FRONTEND_DIST_PROVENANCE_OK ${Object.keys(current.files).length} files`);
} else {
  await writeFile(manifestPath, `${JSON.stringify(current, null, 2)}\n`);
  console.log(`FRONTEND_DIST_PROVENANCE_WRITTEN ${Object.keys(current.files).length} files`);
}
