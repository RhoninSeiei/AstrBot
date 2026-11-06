import assert from 'node:assert/strict';
import test from 'node:test';
import { defaultContextWindow } from '../src/utils/providerContextDefaults.mjs';

test('Codex model creation keeps auto mode instead of copying API metadata', () => {
  const metadata = { limit: { context: 1050000 } };
  assert.equal(defaultContextWindow({ type: 'openai_oauth_chat_completion' }, metadata), 0);
  assert.equal(defaultContextWindow({ auth_mode: 'openai_oauth' }, metadata), 0);
  assert.equal(metadata.limit.context, 1050000);
});

test('other provider defaults preserve valid metadata and unknown models stay automatic', () => {
  assert.equal(defaultContextWindow({ type: 'openai_chat_completion' }, { limit: { context: 128000 } }), 128000);
  assert.equal(defaultContextWindow({ type: 'grok_oauth_chat_completion' }, { limit: { context: 256000 } }), 256000);
  assert.equal(defaultContextWindow({}, null), 0);
  for (const value of [NaN, Infinity, -1, 0, '128000']) {
    assert.equal(defaultContextWindow({}, { limit: { context: value } }), 0);
  }
});
