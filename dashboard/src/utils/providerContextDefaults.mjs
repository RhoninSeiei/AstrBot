/** Keep new Codex providers in request-local automatic context mode. */
export function defaultContextWindow(source, metadata) {
  if (source?.type === 'openai_oauth_chat_completion' || source?.auth_mode === 'openai_oauth') {
    return 0;
  }
  const window = metadata?.limit?.context;
  return typeof window === 'number' && Number.isFinite(window) && window > 0 ? window : 0;
}
