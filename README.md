# TransformerLens (ESM models Implementation Fork)

**Built with ESM.**

This is a research fork of [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), adding support for EvolutionaryScale's protein language models (ESM-3 and ESMC) within the `HookedTransformer` framework.

### Supported Models:
* **ESMC 300M**: Licensed under the [EvolutionaryScale Cambrian Open License](./LICENSE.ESM_OPEN).
* **ESM-3 (Open weights) & ESMC 600M**: Licensed under the [EvolutionaryScale Cambrian Non-Commercial License](./LICENSE.ESM).

---

## License & Attribution

This project is multi-licensed to reflect the different requirements of its components:

1. **TransformerLens Core**: All original library code, infrastructure, and non-ESM related models are licensed under the [MIT License](./LICENSE).
2. **ESMC 300M Integration**: All code and model usage related to ESMC 300M (including `HookedESMC` for the 300M variant) are subject to the [EvolutionaryScale Cambrian Open License](./LICENSE.ESM_OPEN).
3. **ESM-3 & ESMC 600M Integration**: All code and model usage related to ESM-3 and ESMC 600M (including `HookedESM3` and `HookedESMC` for the 600M variant) are subject to the [EvolutionaryScale Cambrian Non-Commercial License](./LICENSE.ESM). **Usage of these specific models is strictly restricted to Non-Commercial Purposes only.**

**Attribution Requirements:**
* Any research, software, or media using this fork must prominently display: **"Built with ESM"**.
* If using **ESMC 300M** for drug discovery or biological target identification, you must also include the attribution: **"Generated with ESM"**.
* Note: Drug discovery and target identification are NOT permitted under the Non-Commercial license for ESM-3 and ESMC 600M.

---

## Citation

If you use this fork in your research, please cite both the original TransformerLens library and the ESM-3/ESMC models:

```BibTeX
@misc{nanda2022transformerlens,
    title = {TransformerLens},
    author = {Neel Nanda and Joseph Bloom},
    year = {2022},
    howpublished = {\url{https://github.com/TransformerLensOrg/TransformerLens}},
}
```BibTeX
@misc{evolutionaryscale2024esm3,
    title={ESM3: Simulating 500 million years of evolution with a language model},
    author={EvolutionaryScale},
    year={2024},
    url={[https://www.evolutionaryscale.ai/blog/esm3-simulating-500-million-years-of-evolution-with-a-language-model](https://www.evolutionaryscale.ai/blog/esm3-simulating-500-million-years-of-evolution-with-a-language-model)}
}
