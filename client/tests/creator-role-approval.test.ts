import assert from "node:assert/strict";
import test from "node:test";
import {
  approveCoordinatorRequest,
  canApproveCoordinatorRequest,
  coordinatorApprovalPrompt,
  type PairingSnapshot,
  type PeerNode,
} from "../src/pairing";

const requester = (overrides: Partial<PeerNode> = {}): PeerNode => ({
  id: "peer-a",
  hostname: "living-room-gpu",
  gpu: "NVIDIA GPU",
  role: "worker",
  self: false,
  stage: "joined",
  online: true,
  wantsCoordinator: true,
  deviceIdentity: "12ab-34cd-56ef",
  ...overrides,
});

const snapshot = (
  peer: PeerNode | null = requester(),
  overrides: Partial<PairingSnapshot> = {},
): PairingSnapshot => ({
  code: "ABC234",
  peers: peer ? [peer] : [],
  coordinatorId: "creator",
  phase: "idle",
  api: null,
  source: "engine",
  isCreator: true,
  ...overrides,
});

test("only the creator sees an online current request with stable identity", () => {
  const peer = requester();
  assert.equal(canApproveCoordinatorRequest(snapshot(peer), peer), true);
  assert.equal(canApproveCoordinatorRequest(snapshot(peer, { isCreator: false }), peer), false);
  assert.equal(canApproveCoordinatorRequest(snapshot(peer, { isCreator: undefined }), peer), false);
  assert.equal(canApproveCoordinatorRequest(snapshot(peer, { phase: "ready" }), peer), false);
  assert.equal(canApproveCoordinatorRequest(snapshot(requester({ wantsCoordinator: false })), requester({ wantsCoordinator: false })), false);
  assert.equal(canApproveCoordinatorRequest(snapshot(requester({ online: false })), requester({ online: false })), false);
  assert.equal(canApproveCoordinatorRequest(snapshot(requester({ deviceIdentity: undefined })), requester({ deviceIdentity: undefined })), false);
  assert.equal(canApproveCoordinatorRequest(snapshot(null), undefined), false);
});

test("the second confirmation names the machine and device and states the real risk", async () => {
  const peer = requester();
  const calls: string[] = [];
  const provider = { async setCoordinator(id: string) { calls.push(id); } };
  let prompt = "";
  const outcome = await approveCoordinatorRequest(provider, snapshot(peer), peer.id, (message) => {
    prompt = message;
    return true;
  });

  assert.equal(outcome, "approved");
  assert.deepEqual(calls, [peer.id]);
  assert.equal(prompt, coordinatorApprovalPrompt(peer));
  assert.match(prompt, /living-room-gpu/);
  assert.match(prompt, /12ab-34cd-56ef/);
  assert.match(prompt, /prompts and responses in plaintext/i);
  assert.match(prompt, /controls the cluster/i);
  assert.match(prompt, /does not prove official software or honest hardware/i);
});

test("cancelling confirmation never invokes the native command", async () => {
  let calls = 0;
  const provider = { async setCoordinator() { calls += 1; } };
  const result = await approveCoordinatorRequest(provider, snapshot(), "peer-a", () => false);
  assert.equal(result, "cancelled");
  assert.equal(calls, 0);
});

test("a withdrawal racing the confirmation is surfaced by the native recheck", async () => {
  const provider = {
    async setCoordinator() {
      throw new Error("[PAIR_ROLE_NOT_REQUESTED] request was withdrawn");
    },
  };
  await assert.rejects(
    approveCoordinatorRequest(provider, snapshot(), "peer-a", () => true),
    /PAIR_ROLE_NOT_REQUESTED/,
  );
});

test("withdrawn, departed, expired, legacy and non-creator requests fail closed", async () => {
  let calls = 0;
  let confirmations = 0;
  const provider = { async setCoordinator() { calls += 1; } };
  const confirm = () => { confirmations += 1; return true; };
  const cases: Array<[string, PairingSnapshot]> = [
    ["withdrawn", snapshot(requester({ wantsCoordinator: false }))],
    ["departed", snapshot(null)],
    ["expired", snapshot(requester({ online: false }))],
    ["legacy", snapshot(requester({ deviceIdentity: undefined }))],
    ["non-creator", snapshot(requester(), { isCreator: false })],
  ];

  for (const [name, current] of cases) {
    assert.equal(
      await approveCoordinatorRequest(provider, current, "peer-a", confirm),
      "stale",
      name,
    );
  }
  assert.equal(confirmations, 0, "stale requests never reach a misleading confirmation");
  assert.equal(calls, 0, "stale requests never cross the native boundary");
});
