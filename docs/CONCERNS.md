# Prompt Eval Concern Register

Wiki home: http://localhost:8088/index.php/Project_Wiki

## Open Concerns

| ID | Concern | Severity | Current mitigation | Next evidence/action |
|---|---|---:|---|---|
| PE-PORT-001 | Infrastructure can look abstract without an applied experiment. | High | Portfolio summary defines the applied experiment-note shape. | Publish one populated experiment packet from a downstream project. |
| PE-PORT-002 | Small case sets can be overgeneralized. | High | Validation register requires frozen case-set scope and failure cases. | Include inclusion criteria and non-covered cases in each packet. |
| PE-PORT-003 | Lightweight statistical comparison can be overclaimed. | Medium | ADR 0005 documents inferential limits. | Use paired-by-input mode where appropriate and state caveats. |
| PE-PORT-004 | Prompt-eval semantics can drift into runtime ownership. | Medium | Capability decomposition assigns runtime to `llm_client`. | Keep execution/observability features in `llm_client`. |
| PE-PORT-005 | Truth-surface pilot evidence can be mistaken for package ownership. | Medium | README says the pilot is historical consumer evidence. | Reopen boundary explicitly before package-owned truth-surface work. |

## Portfolio Judgment

`prompt_eval` is valuable supporting evidence for governed AI engineering. It
should not lead the portfolio by itself, but it strengthens applied projects
when it shows a prompt, schema, or model change was measured on frozen cases
before being adopted.
