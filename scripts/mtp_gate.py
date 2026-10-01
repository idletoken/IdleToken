#!/usr/bin/env python3
"""Real-engine MTP smoke with live-drafting evidence and negative controls.

Run against an isolated engine started with the product's admitted MTP flags.
This never starts/stops a user's process. /completion tests the speculative path
without requesting logprobs; a separate teacher-forced request scores its first
divergence. Logprobs requests deliberately use the exact non-drafting API path.
"""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.request
import zlib


NGRAM_MATCH = 24
DEPTH = 3
PROB_GUARD = 'speculative drafting disabled for this request: token probabilities require target sampling'
IMAGE_GUARD = 'model-based drafting disabled for this request: image embeddings require target-only decoding'
RELEASE = re.compile(r'slot\s+release:.*?task\s+(\d+).*?stop processing')
LAUNCH = re.compile(r'slot\s+launch_slot_:.*?task\s+(\d+).*?processing task')
TOTALS = re.compile(r'draft acceptance\s*=.*?\(\s*(\d+) accepted /\s*(\d+) generated\)')


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def within_band(top, token, limit=0.5):
    scores = {v['id']: v['logprob'] for v in top
              if isinstance(v.get('logprob'), (int, float)) and math.isfinite(v['logprob'])}
    return bool(scores) and token in scores and max(scores.values()) - scores[token] <= limit


def short_budget(prompt_length):
    # Include the full draft depth in the upper bound, even near n_predict.
    budget = min(10, NGRAM_MATCH - prompt_length - DEPTH - 1)
    check(budget >= 4 and prompt_length + budget + DEPTH < NGRAM_MATCH,
          'Fixture cannot exclude ngram-mod')
    return budget


def audit_result(segment, path, payload, result, mtp):
    releases = RELEASE.findall(segment)
    check(len(releases) == 1 and LAUNCH.findall(segment) == releases,
          'Missing or concurrent task invalidates log attribution')
    timing = result.get('timings', {})
    proposed, accepted = timing.get('draft_n', 0), timing.get('draft_n_accepted', 0)
    check(isinstance(proposed, int) and isinstance(accepted, int) and 0 <= accepted <= proposed,
          'Invalid draft counters')
    check(TOTALS.findall(segment) == ([(str(accepted), str(proposed))] if proposed else []),
          'Response and fresh engine draft totals disagree')
    if mtp and path == '/completion':
        kinds = result.get('generation_settings', {}).get('speculative.types', '')
        check(set(kinds.split(',')) - {'none'} == {'ngram-mod', 'draft-mtp'},
              'Unexpected effective speculators')
    if payload.get('n_probs', 0) > 0:
        check(PROB_GUARD in segment and proposed == accepted == 0,
              'Logprobs did not use logged ordinary target decoding')
    media = any(isinstance(m.get('content'), list) and
                any(p.get('type') == 'image_url' for p in m['content'])
                for m in payload.get('messages', []))
    if media:
        check(IMAGE_GUARD in segment and proposed == accepted == 0,
              'Image did not use logged ordinary target decoding')
    return {'task_id': releases[0], 'proposed': proposed, 'accepted': accepted}


def green_png():
    # Same deterministic fixture as the installed Windows gate; no image library.
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data) & 0xffffffff)
    raw = b''.join(b'\0' + b'\0\xff\0' * 96 for _ in range(96))
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!IIBBBBB', 96, 96, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')


def verify_files(path, descriptor):
    """Verify the catalog's whole asset, including every additional shard."""
    check(descriptor.get('repo') and re.fullmatch(r'[0-9a-f]{40}', descriptor.get('revision', '')),
          'Asset source is not pinned to an immutable revision')
    path = Path(path)
    first_name = Path(descriptor['gguf']).name
    check(path.name.endswith(first_name), 'Asset filename does not match its catalog descriptor')
    prefix = path.name[:-len(first_name)]
    parts = descriptor.get('parts', [])
    total = descriptor.get('bytes')
    first_bytes = total - sum(p['bytes'] for p in parts) if total is not None else None
    files = [(path, descriptor['sha256'], first_bytes)]
    files += [(path.parent / (prefix + Path(p['file']).name), p['sha256'], p['bytes']) for p in parts]
    split = re.search(r'-00001-of-(\d{5})\.gguf$', first_name)
    check(not split or int(split[1]) == len(files), 'Catalog does not list every asset shard')
    if split:
        for index, (file, _, _) in enumerate(files, 1):
            check(file.name.endswith(first_name.replace('-00001-of-', f'-{index:05d}-of-')),
                  'Asset shards are not complete and ordered')
    records = []
    for file, expected, size in files:
        check(re.fullmatch(r'[0-9a-f]{64}', expected or '') is not None, 'Missing pinned asset hash')
        check(file.is_file(), 'Missing asset: ' + file.name)
        check(size is None or size > 0 and file.stat().st_size == size, 'Asset byte count differs: ' + file.name)
        digest = hashlib.sha256()
        with file.open('rb') as stream:
            for data in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(data)
        check(digest.hexdigest() == expected, 'Asset hash differs: ' + file.name)
        records.append({'file': file.name, 'bytes': file.stat().st_size, 'sha256': expected})
    return {'repo': descriptor.get('repo'), 'revision': descriptor.get('revision'), 'files': records}


def verify_assets(args):
    check(args.catalog_model and re.fullmatch(r'[a-z0-9.-]+', args.catalog_model),
          'Explicit dependencies require a curated catalog model ID')
    check(args.quant and args.target, 'Catalog quant and target path are required')
    catalog = Path(__file__).resolve().parent.parent / 'models' / (args.catalog_model + '.json')
    model = json.loads(catalog.read_text())
    variants = [v for v in model['variants'] if v['quant'] == args.quant]
    check(len(variants) == 1, 'Quant is not in the selected catalog model')
    variant = variants[0]
    result = {'model': args.catalog_model, 'quant': args.quant,
              'target': verify_files(args.target, variant)}
    if args.draft:
        check(variant.get('mtp_draft_compatible') is True and not variant.get('mtp_layers'),
              'This precision does not admit the separate draft; embedded MTP takes precedence')
        check(model.get('mtp_draft'), 'Catalog has no compatible independent draft')
        result['draft'] = verify_files(args.draft, model['mtp_draft'])
    if args.mmproj:
        check(model.get('mmproj'), 'Catalog has no vision dependency')
        result['mmproj'] = verify_files(args.mmproj, model['mmproj'])
    (args.output / 'verified-assets.json').write_text(json.dumps(result, indent=2) + '\n')
    print('MTP_GATE_ASSETS_OK')


class Fixtures(unittest.TestCase):
    def test_ngram_exclusion(self):
        self.assertLess(9 + short_budget(9) + DEPTH, NGRAM_MATCH)
        with self.assertRaises(AssertionError):
            short_budget(20)

    def test_task_and_guard_controls(self):
        segment = 'slot launch_slot_: id 0 | task 5 | processing task\nslot release: id 0 | task 5 | stop processing\n'
        result = {'timings': {}, 'generation_settings': {'speculative.types': 'none,ngram-mod,draft-mtp'}}
        self.assertEqual(audit_result(segment, '/completion', {}, result, True)['accepted'], 0)
        for bad in ('', segment + segment):
            with self.assertRaises(AssertionError):
                audit_result(bad, '/completion', {}, result, True)
        for payload, guard in (({'n_probs': 10}, PROB_GUARD),
                ({'messages': [{'content': [{'type': 'image_url'}]}]}, IMAGE_GUARD)):
            with self.assertRaises(AssertionError):
                audit_result(segment, '/completion', payload, result, True)
            audit_result(segment + guard, '/completion', payload, result, True)

    def test_split_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / 'model-head-00001-of-00002.gguf', root / 'model-head-00002-of-00002.gguf'
            a.write_bytes(b'first'); b.write_bytes(b'second')
            descriptor = {'gguf': 'head-00001-of-00002.gguf', 'bytes': 11,
                'repo': 'test/fixture', 'revision': '0' * 40,
                'sha256': hashlib.sha256(b'first').hexdigest(),
                'parts': [{'file': 'head-00002-of-00002.gguf', 'bytes': 6,
                           'sha256': hashlib.sha256(b'second').hexdigest()}]}
            self.assertEqual(len(verify_files(a, descriptor)['files']), 2)
            b.write_bytes(b'broken')
            with self.assertRaises(AssertionError):
                verify_files(a, descriptor)
            b.unlink()
            with self.assertRaises(AssertionError):
                verify_files(a, descriptor)
            with self.assertRaises(AssertionError):
                verify_files(a, {**descriptor, 'parts': []})

    def test_catalog_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, draft = root / 'target.gguf', root / 'draft.gguf'
            target.write_bytes(b'target'); draft.write_bytes(b'draft')
            common = {'repo': 'test/fixture', 'revision': '0' * 40}
            variant = {**common, 'quant': 'Q4', 'gguf': target.name,
                       'sha256': hashlib.sha256(b'target').hexdigest(),
                       'mtp_draft_compatible': True, 'mtp_layers': 0}
            model = {'variants': [variant], 'mtp_draft': {**common, 'gguf': draft.name,
                'bytes': 5, 'sha256': hashlib.sha256(b'draft').hexdigest()}}
            args = SimpleNamespace(catalog_model='fixture', quant='Q4', target=target,
                                   draft=draft, mmproj=None, output=root)
            with patch.object(Path, 'read_text', return_value=json.dumps(model)):
                verify_assets(args)
            for incompatible in ({**variant, 'mtp_draft_compatible': False},
                                 {**variant, 'mtp_layers': 1}):
                with patch.object(Path, 'read_text', return_value=json.dumps({**model, 'variants': [incompatible]})):
                    with self.assertRaises(AssertionError):
                        verify_assets(args)

    def test_distribution_negative_control(self):
        self.assertFalse(within_band([{'id': 1, 'logprob': -0.1}, {'id': 2, 'logprob': -9}], 2))
        self.assertFalse(within_band([{'id': 1, 'logprob': float('nan')}], 1))
        self.assertTrue(green_png().startswith(b'\x89PNG\r\n\x1a\n'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:18997')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--quality-only', action='store_true')
    parser.add_argument('--engine-log', type=Path)
    parser.add_argument('--vision', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--verify-assets', action='store_true')
    parser.add_argument('--catalog-model')
    parser.add_argument('--quant')
    parser.add_argument('--target', type=Path)
    parser.add_argument('--draft', type=Path)
    parser.add_argument('--mmproj', type=Path)
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Fixtures))
        return 0 if result.wasSuccessful() else 1
    check(args.output is not None, '--output is required')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.verify_assets:
        verify_assets(args)
        return 0
    check(args.quality_only or args.engine_log and args.engine_log.is_file(),
          'MTP acceptance requires the owned engine log')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    records, short_proofs = [], []

    def post(name, path, payload):
        start = args.engine_log.stat().st_size if args.engine_log else 0
        (args.output / (name + '-request.json')).write_text(json.dumps(payload, indent=2) + '\n')
        request = urllib.request.Request(args.base + path,
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with opener.open(request, timeout=300) as response:
            result = json.load(response)
        (args.output / (name + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        check('error' not in result, f'{name}: engine error')
        if args.engine_log and path in ('/completion', '/v1/chat/completions'):
            deadline = time.monotonic() + 5
            while True:
                check(args.engine_log.stat().st_size >= start, 'Engine log was truncated')
                with args.engine_log.open('rb') as stream:
                    stream.seek(start)
                    segment = stream.read().decode('utf-8', 'replace')
                if RELEASE.search(segment):
                    break
                check(time.monotonic() < deadline, name + ': no fresh task release')
                time.sleep(0.05)
            (args.output / (name + '-engine.log')).write_text(segment)
            records.append({'request': name, **audit_result(segment, path, payload, result, not args.quality_only)})
            (args.output / 'request-attribution.json').write_text(json.dumps(records, indent=2) + '\n')
        return result

    def short_mtp(stage):
        attempts = []
        for index, prompt in enumerate(('def add(a, b):\n    return',
                'The days of the week are Monday,', 'The sequence is 1, 2, 3,')):
            name = f'short-mtp-{stage}-{index}'
            tokens = post(name + '-tokenize', '/tokenize', {'content': prompt, 'add_special': True})['tokens']
            if len(tokens) + DEPTH + 4 >= NGRAM_MATCH:
                attempts.append({'request': name, 'prompt_tokens': len(tokens),
                                 'skipped': 'tokenized fixture exceeds the ngram exclusion bound'})
                continue
            budget = short_budget(len(tokens))
            result = post(name, '/completion', {'prompt': tokens, 'n_predict': budget,
                'temperature': 0, 'seed': 1234, 'return_tokens': True, 'cache_prompt': False})
            check(result.get('tokens_evaluated') == len(tokens) and
                  result.get('generation_settings', {}).get('n_predict') == budget,
                  'Actual prompt/output cap differs from ngram exclusion proof')
            check(result.get('timings', {}).get('cache_n') == 0, 'Short proof reused a prompt')
            counts = records[-1]
            passed = counts['proposed'] > 0 and counts['accepted'] > 0
            attempts.append({'request': name, 'prompt_tokens': len(tokens), 'output_cap': budget,
                'depth_reserve': DEPTH, 'strict_upper_bound': len(tokens) + budget + DEPTH,
                'ngram_n_match': NGRAM_MATCH, 'proposed': counts['proposed'], 'accepted': counts['accepted']})
            proof = {'stage': stage, 'passed': passed, 'attempts': attempts,
                     'attribution': 'MTP by excluding ngram below its minimum match length'}
            (args.output / ('short-mtp-' + stage + '.json')).write_text(json.dumps(proof, indent=2) + '\n')
            if passed:
                short_proofs.append(proof)
                return
        raise AssertionError('No accepted MTP draft under ngram exclusion bound: ' + stage)

    if not args.quality_only:
        short_mtp('initial')

    prompt = 'Write a Python function merging two sorted integer lists. Include type hints and explain the loop.\n\ndef merge_sorted('
    tokens = post('prompt', '/tokenize', {'content': prompt, 'add_special': True})['tokens']
    request = {'prompt': tokens, 'n_predict': 128, 'temperature': 0, 'seed': 1234,
               'return_tokens': True, 'cache_prompt': False}
    baseline = post('baseline', '/completion', {**request, 'n_probs': 10})
    if args.quality_only:
        post('warmup', '/completion', request)
    candidate = post('mtp', '/completion', request)
    timing = candidate.get('timings', {})
    check(timing.get('draft_n', 0) > 0, 'no live draft proposals: this cannot validate MTP')
    check(timing.get('draft_n_accepted', 0) > 0, 'no accepted draft tokens')
    a, b = baseline.get('tokens', []), candidate.get('tokens', [])
    check(a and b, 'missing generated token IDs')
    divergence = next((i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), None)
    if divergence is not None:
        scored = post('divergence', '/completion', {**request,
            'prompt': tokens + b[:divergence], 'n_predict': 1, 'n_probs': 20})
        rows = scored.get('completion_probabilities', [])
        check(rows and within_band(rows[0].get('top_logprobs', []), b[divergence]),
              'first divergent token lies outside target near-tie band')
    else:
        check(len(a) == len(b), 'token streams only share a truncated prefix')
    check(not within_band([{'id': 1, 'logprob': -0.1}, {'id': 2, 'logprob': -9}], 2),
          'negative quality control failed')
    if args.quality_only:
        summary = {'passed': True, 'draft_n': timing['draft_n'],
                   'draft_n_accepted': timing['draft_n_accepted'],
                   'first_divergence': divergence, 'negative_control': 'rejected'}
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('G_SPEC_FAITHFUL_OK ' + json.dumps(summary))
        return

    # Warm the same prefix first: cold n-gram tables previously hid the common
    # accepted-token logprobs defect. The API must return real probabilities.
    logged = post('logprobs-warm', '/completion', {**request, 'n_probs': 10})
    probs = logged.get('completion_probabilities', [])
    check(probs and any(math.isfinite(x['logprob']) and x['logprob'] < -1e-6 for x in probs),
          'logprobs are missing or all zero')
    check(logged.get('timings', {}).get('draft_n', 0) == 0,
          'logprobs request reached unsupported speculative probability path')

    common = {'temperature': 0, 'seed': 7, 'max_tokens': 1024,
              'reasoning_budget_tokens': 128,
              'reasoning_budget_message': 'Time to answer. Give the final answer now.'}
    think = post('reasoning', '/v1/chat/completions', {**common,
        'messages': [{'role': 'user', 'content': 'What is 17 + 25? Explain briefly, then give the answer.'}]})
    msg = think['choices'][0]['message']
    check(bool((msg.get('reasoning_content') or '').strip()), 'reasoning was not separated')
    check(bool((msg.get('content') or '').strip()), 'answer remained inside reasoning')
    tool = post('tool', '/v1/chat/completions', {**common,
        'messages': [{'role': 'user', 'content': 'Get the weather in Paris using the tool.'}],
        'tools': [{'type': 'function', 'function': {'name': 'get_weather',
            'description': 'Get weather for a city', 'parameters': {'type': 'object',
                'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}],
        'tool_choice': 'required'})
    calls = tool['choices'][0]['message'].get('tool_calls', [])
    check(calls and calls[0]['function']['name'] == 'get_weather', 'tool call lost')
    check(json.loads(calls[0]['function']['arguments']).get('city'), 'tool arguments lost')
    schema = post('schema', '/v1/chat/completions', {**common,
        'messages': [{'role': 'user', 'content': 'Return a JSON object with status set to ready.'}],
        'response_format': {'type': 'json_schema', 'json_schema': {'name': 'status',
            'strict': True, 'schema': {'type': 'object',
                'properties': {'status': {'type': 'string', 'enum': ['ready']}},
                'required': ['status'], 'additionalProperties': False}}}})
    check(json.loads(schema['choices'][0]['message']['content']) == {'status': 'ready'},
          'structured output failed')
    image_answer = None
    if args.vision:
        short_mtp('before-image')
        png = green_png()
        (args.output / 'green-fixture.png').write_bytes(png)
        response = post('image', '/v1/chat/completions', {**common,
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': 'What is the single solid color of this image? Answer with the color name only.'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png).decode('ascii')}}]}]})
        image_answer = response['choices'][0]['message'].get('content') or ''
        check(re.fullmatch(r'[\s.*]*(?:the (?:color|colour) is )?green[\s.!*]*', image_answer, re.I) is not None,
              'Image color answer was incorrect')
        after_tokens = post('after-image-tokenize', '/tokenize', {
            'content': 'Write a Python function to reverse a list without changing the input.\n\ndef reversed_list(',
            'add_special': True})['tokens']
        after = post('text-after-image', '/completion', {**request, 'prompt': after_tokens, 'cache_prompt': True})
        check(after.get('timings', {}).get('cache_n') == 0 and
              after.get('timings', {}).get('draft_n_accepted', 0) > 0,
              'Text did not resume drafting with a clean prefix after image')
        short_mtp('after-image')

    summary = {'passed': True, 'draft_n': timing['draft_n'],
               'draft_n_accepted': timing['draft_n_accepted'], 'first_divergence': divergence,
               'long_counts': 'combined ngram/MTP; not per-method acceptance',
               'mtp_proofs': short_proofs, 'vision_tested': args.vision, 'image_answer': image_answer,
               'checks': ['live drafts', 'near-tie quality', 'negative control',
                          'warm logprobs', 'reasoning split', 'tool arguments', 'JSON schema'],
               'scope': 'functional smoke; not a performance benchmark or full PPL gate'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('G_MTP_OK ' + json.dumps(summary))


if __name__ == '__main__':
    raise SystemExit(main())
