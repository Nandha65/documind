# DocuMind Project Sync: Meeting Notes

**Date:** November 4, 2026, 3:00 PM IST
**Attendees:** Priya Raman (core architect, project lead), Marcus Lee (backend engineer), Sofia Almeida (product designer), Daniel Okafor (Head of Security)
**Notes taken by:** Sofia Almeida

## Updates

- **Faster answers (Priya):** Answers now stream into the chat as they are generated, so the first words appear in about a second instead of after the whole answer is ready. The Gemini thinking level was lowered to "low" because simple document questions do not need deep reasoning.
- **Adoption (Sofia):** 46 employees used DocuMind in October, up from 19 in September. The most common topics were leave policy, expenses and laptop setup.
- **Feedback (Sofia):** The top request is to show which document an answer came from. Several people also asked why newly added documents do not show up in answers.

## Decisions

1. Add Anthropic Claude as a third provider alongside Google Gemini and OpenAI. Claude will use local embeddings, because Anthropic does not offer an embeddings API.
2. DocuMind stays approved for Public and Internal documents only. Daniel confirmed that this matches the IT security policy.
3. The index will keep rebuilding only when the chroma_db folder is deleted, until automatic re-indexing is designed.

## Action items

| Owner | Task | Due |
| --- | --- | --- |
| Marcus Lee | Add PDF support (install the PDF reader and test with scanned and text PDFs) | November 20, 2026 |
| Sofia Almeida | Show source documents under each answer | November 27, 2026 |
| Priya Raman | Evaluate chat memory so follow-up questions keep context; present findings at the next sync | November 18, 2026 |
| Daniel Okafor | Security review of the Claude integration | November 18, 2026 |

## Open questions

- Should DocuMind re-index automatically when files in the data folder change?
- Do we need per-team document collections, or is one shared collection enough for now?

**Next meeting:** November 18, 2026, 3:00 PM IST.
