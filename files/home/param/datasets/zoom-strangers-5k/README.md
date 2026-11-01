---
license: other
language:
- en
task_categories:
- text-generation
pretty_name: zoom-strangers-5k
---

# zoom-strangers-5k

Text-only transcripts of first Zoom calls between two complete strangers who know nothing about each other, not even a name.
Both are normal, everyday people: the `assistant` side is friendly and good at keeping a conversation going without being a performer;
the `user` side is an ordinary person whose manner varies (reserved, neutral, warm but low-key, a bit nervous, skeptical, distracted, fairly chatty).
The user always opens the call; every row ends on the assistant.

- Splits: `train.jsonl` (4900) and `eval.jsonl` (100, 2% stratified by setting).
- Rows: `{"id", "messages": [{"role": "user"|"assistant", "content"}], "meta"}`; no system prompt. `meta` holds the setting, both character cards, the ordinary person's manner, the planned Zoom glitch, judge scores and models.
- Settings (about 500 each): virtual coffee roulette, networking-event breakout, online class breakout, virtual-conference hallway, language-exchange match, hobby-club social night, volunteer pairing, speed networking, workshop pair exercise, coworking break room. All are anonymous random pairings.

## How it was made
- A planner (z-ai/glm-5.3-flash, MIT) writes each call's setting and one character card per person; neither card mentions the other person. Names, jobs and home towns are dealt from shuffled decks so they spread evenly.
- Turn by turn: the friendly person is played by xiaomi/mimo-v2.6-flash (MIT) and the ordinary person by z-ai/glm-5.3-flash (MIT). Each sees only its own card, the kind of call it joined and the conversation so far.
- Spoken style: plain everyday words, numbers in words, no markdown, symbols or stage directions, turns of at most ninety words (mostly under forty). Turns are regenerated for digits or symbols, overlong turns, or using the other person's name before it was said.
- Wrap-up starts at the word target (1,000 to 1,500 words, at most 1,330); the first goodbye ends the call and the other person replies once.
- Kept only if openai/gpt-oss-120b (Apache-2.0) scores natural, realism, engaging-but-normal, ordinary, strangers, coherence, ending and turn-taking all at least 4 of 5, and the call is 900 to 1,700 words.

## Stats
```
{
 "attempted": 6830,
 "kept": 5059,
 "final": 5000,
 "eval": 100,
 "train": 4900,
 "rejected": 1639,
 "errors": {
  "KeyError('glitch')": 31,
  "RuntimeError('turn failed: symbols_or_digits')": 61,
  "RuntimeError('turn failed: too_long')": 10,
  "RuntimeError('turn failed: empty')": 23,
  "RuntimeError('turn failed: emoji_or_script')": 4,
  "RuntimeError('turn failed: name_leak')": 2,
  "RuntimeError('z-ai/glm-5.3-flash: no answer')": 1
 },
 "spend_this_process": {
  "usd": 7.552281713200009,
  "calls": 63796,
  "errors": 0
 },
 "by_setting_kept": {
  "coffee_roulette": 510,
  "networking_breakout": 511,
  "class_breakout": 501,
  "conference_hallway": 502,
  "language_exchange": 513,
  "hobby_club_social": 510,
  "volunteer_pairing": 483,
  "speed_networking": 516,
  "workshop_pair": 502,
  "coworking_break": 511
 }
}
```
