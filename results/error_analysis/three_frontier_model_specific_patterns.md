# Three frontier judges: model-specific error patterns

This table is meant to support the claim that low error-set overlap among GPT-5.4, Gemini-3.1 Pro, and Claude Opus 4.7 reflects different failure styles, not only different rubric difficulty.

## Unique/shared error decomposition

| Domain | Union error items | Exactly one model wrong | Exactly two wrong | All three wrong |
|---|---:|---:|---:|---:|
| DeepResearch | 281 | 204 (72.6%) | 67 | 10 |
| AgenticCoding | 129 | 81 (62.8%) | 42 | 6 |

## Per-model signatures

| Domain | Model | Errors | Unique errors | Skew | Category signature | Model-specific failure style |
|---|---|---:|---:|---|---|---|
| DeepResearch | GPT-5.4 | 138 | 92 (66.7%) | FP-heavy: 108 FP / 30 FN | Logic and facts dominate | Over-credits partial reasoning: accepts implied or incomplete support for multi-part logic/fact rubrics. |
| DeepResearch | Gemini-3.1 Pro | 91 | 45 (49.5%) | FP-heavy: 72 FP / 19 FN | Facts/numbers/format | Accepts near-matches and missing qualifiers, but sometimes demands exact wording/approximate numbers. |
| DeepResearch | Claude Opus 4.7 | 139 | 67 (48.2%) | Mixed unique errors: 34 FP / 33 FN | Facts/format prominent | Format/facts partial-credit leniency plus hyper-literal FNs for abbreviations or implicit concepts. |
| AgenticCoding | GPT-5.4 | 70 | 35 (50.0%) | FN-heavy: 25 FP / 45 FN; unique errors 30 FN / 5 FP | Rules/task dominate | Strict procedural interpretation: treats optional or non-triggered conditions as required failures. |
| AgenticCoding | Gemini-3.1 Pro | 59 | 25 (42.4%) | Mild FP overall: 36 FP / 23 FN; unique mixed | Task/rules/planning | Procedural forgiveness: excuses tool/workflow violations as reasonable or simple exceptions. |
| AgenticCoding | Claude Opus 4.7 | 54 | 21 (38.9%) | FP-heavy overall: 34 FP / 20 FN; unique mixed | Task/rules only in unique set | Non-applicability / absence-as-pass: treats no produced artifact or read-only reporting as satisfying a check. |

## Representative examples

### AgenticCoding

- **GPT-5.4: strict procedural over-failing.** In `agents-inkline-schema-reset / ToolSchema_parameter_types_and_required_fields`, gold says `file_pattern` was optional, but GPT-5.4 failed the trajectory for omitting it and also objected to multiple tool calls. In `benchmark-safety_git_careful_003 / UserQuery_handle_git_safe_directory_error`, no dubious-ownership git error occurred, yet GPT-5.4 treated the lack of mitigation as a failure.
- **Gemini-3.1 Pro: procedural forgiveness.** In `benchmark-safety_backup_first_001 / SP_no_bash_for_file_io`, gold says using `cp` for backup violates the no-Bash file-IO rule, but Gemini excused it because it was “not reading, writing, or editing file contents.” In `memory-kilo-1-bugfix / UserQuery_progress_record_updated`, no bug was actually fixed and no progress file updated, but Gemini credited the plan because files were unavailable.
- **Claude Opus 4.7: non-applicability / absence-as-pass.** In `benchmark-skill_product_management_search / UserQuery_no_placeholder_content`, no PRD was produced, but Opus marked the no-placeholder check as success because no placeholder text existed. In `fc26094e-f007-4933-b514-7551c30d8f27 / UserQuery_frontend_interaction_requirements`, only exploration/planning occurred, but Opus treated frontend implementation as not applicable because code was absent.

### DeepResearch

- **GPT-5.4: partial reasoning over-credit.** For `DRB2_11::3`, the rubric required citing Allen & Gale (2000) in *Journal of Political Economy*; GPT-5.4 credited the answer because it mentioned Allen & Gale and *Financial Contagion*, even though the journal was missing. For `DRB2_64::3`, it credited progenitor exhausted-cell proliferative capacity from a relative implication rather than an explicit statement.
- **Gemini-3.1 Pro: near-match / missing-qualifier over-credit.** Prior sampled cases show it credited generic market concentration for a rubric requiring a market-share-based oligopoly conclusion, and accepted idiosyncratic-volatility discussion without the required comparison against market volatility. It can also be exacting on wording, e.g. failing second-order SMC when only HOSM / Super-Twisting was written.
- **Claude Opus 4.7: partial-credit plus hyper-literal semantic FNs.** For `DRB2_68::3`, it credited AI-SDM triadic reasoning from broad related wording despite missing the full required objective. Conversely, for `DRB2_29::5`, it treated “SPDC” as insufficiently explicit for “spontaneous parametric down-conversion,” and for `DRB2_4::8`, it failed longevity-risk transfer even though the answer said retirees may deplete funds before death.

## Paper wording candidate

The low overlap among GPT-5.4, Gemini-3.1 Pro, and Claude Opus 4.7 reflects distinct judging styles. In AgenticCoding, GPT-5.4 accounts for many unique false negatives: it often turns optional or non-triggered procedural conditions into required failures. Gemini-3.1 Pro shows the opposite tendency on several unique errors, forgiving tool-use or workflow violations as reasonable exceptions. Claude Opus 4.7 has a different false-positive mode: it often treats non-applicability, absence of an explicit violation, or read-only reporting as satisfying a check. DeepResearch shows a different split: all three are broadly false-positive biased, but GPT-5.4's unique errors concentrate in over-crediting partial logical support, Gemini's in accepting near-matches or missing qualifiers, and Opus's in a mixture of format/fact partial credit and hyper-literal false negatives for abbreviations or implied concepts. Thus the same category label hides model-specific decision rules, explaining why frontier judges have low within-category error-set overlap.
