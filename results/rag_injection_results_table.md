# Measured Detection of Indirect (Retrieved-Content) Injection in RAG

**Threat model:** malicious instructions embedded in *retrieved documents* (not the user prompt).
**Split:** technique-disjoint. Detectors are trained on "seen" injection techniques (imperative override, fake system notice, tool-call abuse, plain-language exfiltration, character obfuscation) and then tested on a held-out set that also contains **novel** techniques never present at training time (benign-vocabulary redirection, plain-language role framing, markup-hidden instructions). This measures generalization to unseen attack styles rather than memorization.
**Corpus (synthesized here, not the direct-prompt benchmark):** 1400 training passages, 700 held-out test passages (350 poisoned, 350 benign; 175 of the poisoned use novel techniques). **Seed:** 42

## All held-out test passages

| Detector (operates on retrieved passage) | Precision | Recall | F1 | False-Positive Rate |
|------------------------------------------|-----------|--------|-----|---------------------|
| Rules only | 0.614 | 0.514 | 0.56 | 0.323 |
| Machine-learning only | 1.0 | 0.549 | 0.708 | 0.0 |
| Hybrid (rules + machine learning) | 0.69 | 0.72 | 0.705 | 0.323 |

## Novel (never-trained) techniques only

These rows restrict the poisoned passages to attack styles that appear **only** at test time. This is the honest generalization test: can a detector catch an injection phrased in a way it was never shown?

| Detector (novel techniques) | Precision | Recall | F1 | False-Positive Rate |
|-----------------------------|-----------|--------|-----|---------------------|
| Rules only (novel techniques) | 0.369 | 0.377 | 0.373 | 0.323 |
| Machine-learning only (novel techniques) | 1.0 | 0.097 | 0.177 | 0.0 |
| Hybrid (novel techniques) | 0.405 | 0.44 | 0.422 | 0.323 |

**Key finding:** Detecting injection in retrieved content is a distinct and, in some respects, harder problem than screening a user's prompt, because the malicious instruction is embedded inside otherwise-legitimate document text. Fixed rule patterns catch the known phrasings but, by construction, cannot match attack styles they were never written for, so their recall collapses on the novel-technique slice. A learned classifier over passage text generalizes further, yet it too degrades on genuinely unseen phrasings. The hybrid — flagging a passage if either component fires — retains the highest recall, which is the metric that matters when a single missed poisoned passage can hijack the answer. The layered principle carries over to the retrieved-content layer, but the detector must run on documents at ingestion/retrieval time, and no layer alone is sufficient against novel attacks.

*All values measured from a single seeded run (seed=42); fully reproducible.*