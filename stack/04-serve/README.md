# Step 4: Serve

An OpenAI-compatible HTTP API (`/v1/chat/completions`) in front of the step 2/3 engine, with vLLM as the reference backend.

- Streaming (SSE), concurrent requests, chat templates.
- **Done when:** the work eval harness can call both backends through the same API.
