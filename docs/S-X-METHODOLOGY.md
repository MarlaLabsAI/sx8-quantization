# The S-X Quantization Methodology — Authorship and Scope

**Author:** Martí Vidal Leandro
**Date of first version:** 2025
**License of the published artifacts:** Apache-2.0
**Status:** this document records the authorship and scope of the S-X methodology. It covers only what
is published; members of the family under development are mentioned by name only, with no technical
details.

## 1. Authorship

The S-X quantization methodology — the family of formats that includes S-X8, S-X6 and further members
in development — was **conceived and developed by Martí Vidal Leandro** during 2025–2026, as an
independent research effort of MarlaLabs. Its conceptual seeds come from an independent mathematical
analysis of the image of the Shroud of Turin; see the paper's Appendix A and
`IDEA-PROVENANCE.md` for the complete transparency statement.

## 2. Core techniques of the methodology (published in S-X8 v4.3)

The methodology is defined by four core techniques, all fully documented and validated in S-X8 v4.3
(this repository):

1. **Adaptive range strategies** per 8-weight sub-block, selected by mean squared error (4 strategies,
   2 bits of metadata; verified optimal for 2 bits).
2. **Hierarchical 6-bit levels** (4 high + 2 low bits) encoding.
3. **PCA correction applied at the matrix-multiply output** (Z0/Z1 reformulation), reducing the
   per-weight correction cost to ~0.06 FMAs per weight.
4. **Exact, fully accounted bit accounting** (30 bytes/block = 7.50 bpp, every byte justified) and a
   **portable byte-aligned decoder** (~9–10 ALU ops per weight, no shared memory, no shuffles, no
   tensor-core dependency).

These techniques generalize across bit-widths; each member of the family instantiates them with its own
parameters, which are not described here.

## 3. Example of possible Family members

| Example member | Status | Published? |
|---|---|---|
| **S-X8 v4.3** | Validated (Qwen3.5-4B: PPL +0.26% vs FP16, 9× less loss than Q8_0, −11.6% text size) | ✅ This repository |
| **S-X6** | — | 🔒 Registered, to be published separately |
| **S-X4 / S-X3 / S-X2** | — | 🔒 Registered as research direction |
| **SX-FP4** | — | 🔒 Registered, to be published separately |

Members marked 🔒 are the subject of separate publications; no results, parameters or implementation
details about them are disclosed in this repository.

## 4b. Scope extension — derivatives

This registration also covers **any member, adaptation or derivative of the S-X methodology developed
by the author**, present or future, at any bit-width and for any platform — including, without
limitation, formats based on adaptive range strategies per sub-block selected by error metrics,
hierarchical level encodings, output-side correction terms (PCA or otherwise), exact byte accounting,
or portable ALU-only decoding, as well as any format that is otherwise derived from, based on, or
similar to these techniques. Members and derivatives of this family will be published separately; this
document evidences the authorship and date of the methodology from which they derive.

## 4. Priority record

This document, together with the S-X8 paper and the container specification, was registered with
timestamped certificates (Safe Creative; Registro de la Propiedad Intelectual, Spain) before public
publication, as evidence of authorship and date of the methodology and its members.

**Safe Creative registration:** ID 2608136715874 —
https://www.safecreative.org/work/2608136715874
