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
@article {hayes2024simulating,
	author = {Hayes, Thomas and Rao, Roshan and Akin, Halil and Sofroniew, Nicholas J. and Oktay, Deniz and Lin, Zeming and Verkuil, Robert and Tran, Vincent Q. and Deaton, Jonathan and Wiggert, Marius and Badkundri, Rohil and Shafkat, Irhum and Gong, Jun and Derry, Alexander and Molina, Raul S. and Thomas, Neil and Khan, Yousuf A. and Mishra, Chetan and Kim, Carolyn and Bartie, Liam J. and Nemeth, Matthew and Hsu, Patrick D. and Sercu, Tom and Candido, Salvatore and Rives, Alexander},
	title = {Simulating 500 million years of evolution with a language model},
	year = {2025},
	doi = {10.1126/science.ads0018},
	URL = {http://dx.doi.org/10.1126/science.ads0018},
	journal = {Science}
}

@misc{esm2024cambrian,
  author = {{ESM Team}},
  title = {ESM Cambrian: Revealing the mysteries of proteins with unsupervised learning},
  year = {2024},
  publisher = {EvolutionaryScale Website},
  url = {https://evolutionaryscale.ai/blog/esm-cambrian},
  urldate = {2024-12-04}
}

@software{evolutionaryscale_2024,
  author = {{EvolutionaryScale Team}},
  title = {evolutionaryscale/esm},
  year = {2024},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.14219303},
  URL = {https://doi.org/10.5281/zenodo.14219303}
}
