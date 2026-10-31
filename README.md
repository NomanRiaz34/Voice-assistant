# Travel Voice — Streamlit

ChatGPT-inspired browser UI for your original OpenAI Agents SDK travel advisor and `VoicePipeline`.

## Install

Requires Python 3.11+ and a working OpenAI API key with access to the selected model, transcription, and speech synthesis.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (do not commit `.env`).

```bash
streamlit run app.py
```

## Streamlit Community Cloud

Deploy `app.py` with `requirements.txt`. In app settings → Secrets add:

```toml
OPENAI_API_KEY = "sk-..."
```

**Do not put your key into a public repository.** A deployed app using the owner's server key should be protected with authentication and appropriate rate limiting / usage limits before being made public.

## What it supports

- Browser mic recording with `st.audio_input` and explicit Send button
- 24 kHz mono PCM conversion before `AudioInput`
- Your original `VoicePipeline` using `gpt-5.4-mini` and default `coral` TTS
- Transcript and spoken answer in chat-style messages
- Browser audio replay, optional autoplay, and session-only multiple chats
- Short recent conversation context in agent instructions

## Limitations

- Turn-based, not continuous live realtime audio; the answer arrives after the server processes the recording.
- History is per Streamlit session; refresh/restart may clear it. No persistent database.
- Browser autoplay may be blocked by browser policy; manual playback remains available.
- Input transcript is captured via `SingleAgentWorkflowCallbacks`, and reply text via `total_output_text`.
- Voice requests can incur API charges. The OpenAI API key is not a ChatGPT subscription credential.
