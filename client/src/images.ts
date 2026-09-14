// Attaching images to a chat turn.
//
// Two representations per picture, and the split is the whole design:
//
//   `full`  — what the model is asked to look at. Sent once, with the turn it
//             belongs to, and never written to storage.
//   `thumb` — what the transcript keeps. A few KB, so a conversation history
//             stays inside the localStorage quota.
//
// Why that matters more than it looks: history is persisted by stringifying
// every conversation into ONE localStorage key. A phone screenshot is a couple
// of MB once base64-encoded, and `setItem` past the quota THROWS — which would
// not lose the picture, it would lose the user's entire chat history, silently,
// at the moment they attached one image too many. Thumbnails keep the write
// small by construction rather than by hoping.
//
// The cost is stated rather than hidden: after a reload, a turn that carried an
// image replays with the thumbnail. The model still sees the picture, at lower
// resolution. The alternative — dropping it — makes the model answer follow-up
// questions about an image it can no longer see, which is the failure this
// whole feature exists to remove.

/** Longest edge of a stored thumbnail, in pixels. Small enough that a dozen fit
 *  in the quota alongside the text, large enough to still read as the picture
 *  it came from. */
const THUMB_MAX_PX = 256;
/** JPEG quality for thumbnails. They are evidence of what was sent, not the
 *  thing being analysed. */
const THUMB_QUALITY = 0.7;

/** Refused above this, with a reason. Not a technical ceiling — the engine
 *  rescales anyway — but base64 of a huge file freezes the webview while it is
 *  built, and "the app hung" is a worse answer than "that file is too big". */
export const MAX_IMAGE_BYTES = 20 * 1024 * 1024;

export interface Attachment {
  /** data: URL of the original bytes. Sent with the turn; never persisted. */
  full: string;
  /** data: URL of the downscaled copy. Persisted and displayed. */
  thumb: string;
  /** For the alt text and the tooltip; not sent to the model. */
  name: string;
}

/** MIME types the engine's vision path accepts. Anything else is refused here
 *  rather than at the far end of the request, where the message would be the
 *  engine's and the context would be gone. */
const ACCEPTED = new Set(["image/png", "image/jpeg", "image/webp", "image/gif"]);

export function isAcceptedImage(type: string): boolean {
  return ACCEPTED.has(type);
}

function readAsDataUrl(file: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onerror = () => reject(new Error("could not read the file"));
    fr.onload = () => resolve(String(fr.result ?? ""));
    fr.readAsDataURL(file);
  });
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onerror = () => reject(new Error("not a readable image"));
    img.onload = () => resolve(img);
    img.src = src;
  });
}

/** Downscale to a thumbnail. Returns the ORIGINAL when it is already smaller
 *  than the target — re-encoding a 64x64 icon through JPEG would make it both
 *  bigger and worse. */
async function makeThumb(dataUrl: string): Promise<string> {
  const img = await loadImage(dataUrl);
  const longest = Math.max(img.naturalWidth, img.naturalHeight);
  if (longest <= THUMB_MAX_PX && dataUrl.length < 64 * 1024) return dataUrl;
  const scale = THUMB_MAX_PX / longest;
  const w = Math.max(1, Math.round(img.naturalWidth * scale));
  const h = Math.max(1, Math.round(img.naturalHeight * scale));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return dataUrl; // no 2d context: keep the original rather than lose it
  // White underneath: a transparent PNG flattened onto JPEG's default black
  // turns a screenshot of a document into an unreadable dark rectangle.
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0, w, h);
  return canvas.toDataURL("image/jpeg", THUMB_QUALITY);
}

/** One picked file -> an attachment, or a thrown Error whose message is
 *  already a sentence for the user. */
export async function attachmentFromFile(file: File): Promise<Attachment> {
  if (!isAcceptedImage(file.type)) {
    throw new Error(`${file.name}: not an image this model can read`);
  }
  if (file.size > MAX_IMAGE_BYTES) {
    throw new Error(
      `${file.name}: ${(file.size / 1048576).toFixed(1)} MB is larger than the ` +
      `${MAX_IMAGE_BYTES / 1048576} MB limit`);
  }
  const full = await readAsDataUrl(file);
  const thumb = await makeThumb(full);
  return { full, thumb, name: file.name };
}

/** Split a `data:<mime>;base64,<payload>` URL for the Anthropic image block,
 *  which wants the two halves separately. Returns null for anything that is not
 *  a base64 data URL — a remote https: image would be passed through as a url
 *  source instead, and a malformed one must not be sent as if it were fine. */
export function splitDataUrl(url: string): { mediaType: string; data: string } | null {
  const m = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  return m ? { mediaType: m[1], data: m[2] } : null;
}
