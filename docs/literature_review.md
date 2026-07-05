# Literature Review: Wasserstein-Robust Q-Learning for Antimicrobial Resistance Surveillance

## Scope and review method

This scoped narrative review examines whether Wasserstein-robust tabular Q-learning provides a defensible methodological basis for population-informed antibiotic-class selection under temporal antimicrobial resistance (AMR) drift. The review was developed in the project's existing NotebookLM notebook. The notebook contained the reference algorithm paper and its software repository; its research function was used to identify additional literature on distributionally robust decision-making, reinforcement learning (RL) for antibiotic stewardship, and the use of European Antimicrobial Resistance Surveillance Network (EARS-Net) data in policy. Sources were classified as peer-reviewed studies, systematic reviews, official surveillance reports, preprints, or software artifacts.

This process supports a focused synthesis, not a systematic review. It did not use a preregistered protocol, duplicate screening, or an exhaustive search of bibliographic databases. Claims of novelty are therefore limited to the reviewed corpus. Preprints and software repositories are used to map emerging work, not as equivalent substitutes for peer-reviewed evidence.

## Distributionally robust reinforcement learning

Classical Q-learning estimates an action-value function under the transition process represented in its training experience. When the transition distribution changes, the learned policy may perform poorly even if training convergence was satisfactory. Distributionally robust Markov decision processes address this problem by optimizing performance against an ambiguity set of plausible transition distributions rather than a single estimated model.

Neufeld and Sester formulate such an ambiguity set as a Wasserstein ball around a reference transition measure [1]. Wasserstein distance is useful when uncertainty concerns movement of probability mass between states because the ground cost encodes the distance between those states. Their robust Q-learning method converts the worst-case expectation into a tractable dual optimization and establishes convergence under the assumptions of the finite-state formulation. The accompanying software demonstrates the method on compact non-clinical examples [2]. This work supplies the algorithmic foundation for the present project, but it does not establish effectiveness for AMR surveillance data.

The robustness radius is not a minor implementation parameter. It determines how far the adversarial transition distribution may depart from the reference kernel. A radius that is too small approximates the nominal solution; one that is too large can produce an excessively conservative policy. Offline distributionally robust RL theory similarly emphasizes that robustness depends on the geometry and size of the uncertainty set, although some recent work studies total-variation rather than Wasserstein ambiguity and linear rather than tabular MDPs [3]. Consequently, the radius should have an empirical interpretation and its policy consequences should be reported across a prespecified range.

## Reinforcement learning in antibiotic stewardship

The antibiotic decision literature contains several related but distinct uses of AI and RL. Clinical RL studies commonly use patient-level trajectories to recommend the timing or dosage of treatment, particularly in sepsis. For example, Wang and colleagues used a deep Q-network with clinically informed rewards to recommend antibiotic combinations and durations from patient data [11]. These models operate on physiological measurements, laboratory observations, treatment histories, and clinical outcomes. Their setting differs fundamentally from country-year surveillance: they attempt to learn individualized sequential decisions, whereas EARS-Net does not contain patient covariates, treatment assignments, or counterfactual outcomes.

Other work studies antibiotic policies in simulated ecological systems. Recent simulation artifacts represent prescribing actions and resistance prevalence as an MDP, allowing an agent to trade immediate treatment effectiveness against longer-term resistance [4]. Related experimental and simulation research investigates treatment strategies intended to limit the evolution of resistance [5]. Such studies are valuable for testing causal hypotheses, but their conclusions depend on assumed biological and behavioral dynamics. They do not validate the claim that an antibiotic recommendation derived from aggregated surveillance will alter subsequent resistance.

Reviews of AI-supported antimicrobial stewardship describe growing interest in prediction and prescription support but also identify heterogeneous data, limited external validation, and uncertain clinical transportability [6]. The appropriate lesson for this project is therefore restraint. Its actions can be interpreted as population-informed class selections evaluated against next-year observed susceptibility. They cannot be interpreted as patient prescriptions, and the action-independent transition model cannot estimate the causal effect of antibiotic use on future resistance.

## EARS-Net surveillance and policy

EARS-Net provides standardized surveillance of invasive bacterial isolates reported by participating European countries. Its longitudinal country-level coverage makes it suitable for studying geographic variation and temporal drift. The 2024 report shows that AMR remains substantial and geographically heterogeneous across the EU/EEA; the estimated incidence of bloodstream infections caused by third-generation cephalosporin-resistant *Escherichia coli* was 5.9% higher in 2024 than in the 2019 baseline year [7]. These observations motivate temporal evaluation rather than random train-test splitting.

The same data source imposes strict limits. EARS-Net primarily represents invasive isolates, and ECDC cautions that these isolates may not represent organisms from urinary, respiratory, wound, or other infections [8]. Differences in sampling, laboratory practice, reporting coverage, case mix, and healthcare systems also complicate cross-country comparison. A country-year state is therefore an epidemiological summary, not a patient state.

The COMBACTE-Magnet EPI-Net COACH review examined how AMR surveillance can inform empirical antibiotic policy [9]. It supports the general relevance of resistance surveillance to stewardship decisions, while showing that recommended thresholds and reporting practices often rest on limited or heterogeneous evidence. This distinction is important: surveillance can inform policy, but it does not by itself identify the best treatment for an individual. In the present project, training-derived binary thresholds are reproducible state-construction devices rather than clinical susceptibility breakpoints.

## Distributional robustness in healthcare

Distributionally robust individualized treatment rules provide a healthcare precedent for optimizing decisions when the deployment population may differ from the training population. Mo and colleagues maximize worst-case value over nearby distributions and evaluate generalizability under covariate shifts [10]. Their work supports the broader rationale for guarding against distribution shift in health-related decisions. It does not, however, solve the same problem: an individualized treatment rule is a static patient-level decision under causal identification assumptions, whereas this project uses an aggregated sequential MDP with action-independent transitions.

This comparison clarifies what Wasserstein robustness can and cannot contribute. It can test whether a policy is less dependent on one estimated historical transition kernel. It cannot compensate for absent patient variables, establish treatment effects, correct all surveillance biases, or convert an ecological association into clinical evidence.

## Research gap and project contribution

Within the reviewed corpus, no study was identified that evaluates Wasserstein-robust tabular Q-learning on country-year EARS-Net *E. coli* resistance states for one-year-ahead antibiotic-class selection. This statement is deliberately narrower than a claim that no such study exists.

The project's novelty lies in combining four elements that the literature generally treats separately:

1. an observed, multi-country AMR surveillance series rather than a wholly simulated resistance environment;
2. a Wasserstein ambiguity set whose ground cost reflects distances between interpretable resistance states;
3. rolling, one-year-ahead evaluation across an explicit pandemic-era temporal shift; and
4. an action-independent formulation that evaluates robustness without claiming that the selected action causes future resistance transitions.

This is an application and evaluation contribution, not a new RL algorithm. Its academic value does not require the robust policy to outperform classical Q-learning. A finding that empirically calibrated Wasserstein balls collapse to a carbapenem-heavy policy, or reduce average adjusted coverage, is informative if the implementation is validated and the sensitivity analysis explains when the behavior appears. The radius must not be selected retrospectively merely to obtain a favorable result. Instead, an empirically motivated primary radius, a prespecified radius grid, policy-switch points, and nominal baselines should be reported together.

## Implications for the study design

The literature supports the project's use of temporal validation, explicit uncertainty sets, and simple baselines. It also suggests several safeguards. First, the evaluation should distinguish raw susceptibility from the carbapenem-adjusted reward so that apparent robustness is not confused with broad-spectrum use. Second, results should be stratified by year and country because average performance can conceal geographic failures. Third, robustness should be interpreted through both performance and policy composition. Fourth, the myopic comparator is essential: with action-independent transitions and action-dependent immediate rewards, a complex sequential learner may have little justified advantage over a transparent state-wise rule.

The strongest conclusion available from this design is methodological: it can show how a published robust RL algorithm behaves when exposed to observed AMR drift under transparent population-level assumptions. It cannot recommend treatment, estimate stewardship effects, or demonstrate improved patient outcomes.

## Evidence limitations

The evidence base is uneven. The core Wasserstein Q-learning method is peer reviewed, but antibiotic-specific RL work includes preprints and simulation software. EARS-Net provides authoritative surveillance, yet its ecological structure does not support patient-level inference. The healthcare distributional-robustness literature demonstrates relevance under shift but usually addresses different estimands and data structures. Finally, this review's NotebookLM-assisted retrieval was scoped rather than exhaustive. A publication manuscript would require a reproducible database search and formal screening process.

## Source classification

| Source | Year | Evidence type | Role in this review | Principal limitation |
| --- | ---: | --- | --- | --- |
| Neufeld and Sester, *Robust Q-learning algorithm for Markov decision processes under Wasserstein uncertainty* | 2024 | Peer-reviewed methods study | Core algorithm, ambiguity set, and convergence basis | Validation is not specific to AMR or healthcare |
| Sester, *Wasserstein-Q-learning* repository | 2022 | Software artifact | Reference implementation and examples | Repository evidence is not independent empirical validation |
| *Sample Complexity of Offline Distributionally Robust Linear Markov Decision Processes* | 2024 | Peer-reviewed methods study | Context for offline robust RL and uncertainty calibration | Linear MDP and total-variation setting differ from this project |
| *abx_amr_simulator* | 2026 | Preprint and software artifact | Emerging antibiotic-policy RL formulation | Simulated dynamics may not represent observed resistance processes |
| Weaver et al., *Reinforcement learning informs optimal treatment strategies to limit antibiotic resistance* | 2024 | Peer-reviewed experimental and simulation study | RL under resistance-evolution dynamics | In-silico drug cycling over empirical fitness landscapes differs from EARS-Net surveillance |
| AI-driven antibiotic stewardship systematic review | 2025 | Systematic review | Broader clinical AI context and evidence gaps | Heterogeneous methods and outcomes; not focused on robust tabular RL |
| Wang et al., *Clinical knowledge-guided deep reinforcement learning for sepsis antibiotic dosing recommendations* | 2024 | Peer-reviewed modeling study | Patient-level antibiotic RL comparator | Retrospective model evaluation does not establish clinical effectiveness |
| ECDC, *Antimicrobial resistance in the EU/EEA: 2024 data* | 2025 | Official surveillance report | AMR burden, temporal change, and geographic heterogeneity | Aggregated invasive-isolate surveillance |
| ECDC, EARS-Net network documentation | current web resource | Official documentation | Scope and representativeness limits of the data | Describes surveillance, not policy effectiveness |
| Tacconelli et al., COMBACTE-Magnet EPI-Net COACH project | 2020 | Systematic review | Link between AMR surveillance and empirical policy | Underlying recommendations often have limited evidence |
| Mo et al., *Learning Optimal Distributionally Robust Individualized Treatment Rules* | 2021 | Peer-reviewed methods study | Healthcare precedent for robustness under distribution shift | Static, patient-level treatment-rule setting |

## References

1. Neufeld A, Sester J. Robust Q-learning algorithm for Markov decision processes under Wasserstein uncertainty. *Automatica*. 2024;168:111825. <https://doi.org/10.1016/j.automatica.2024.111825>
2. Sester J. Wasserstein-Q-learning. GitHub repository. 2022. <https://github.com/juliansester/Wasserstein-Q-learning>
3. Wang H, Shi L, Chi Y. Sample complexity of offline distributionally robust linear Markov decision processes. *Reinforcement Learning Journal*. 2024;3:1467-1510. <https://rlj.cs.umass.edu/2024/papers/Paper189.html>
4. *abx_amr_simulator: A simulation environment for antibiotic prescribing policy optimization under antimicrobial resistance*. arXiv preprint. 2026.
5. Weaver DT, King ES, Maltas J, Scott JG. Reinforcement learning informs optimal treatment strategies to limit antibiotic resistance. *Proceedings of the National Academy of Sciences*. 2024;121(16):e2303165121. <https://doi.org/10.1073/pnas.2303165121>
6. Harandi H, et al. Artificial intelligence-driven approaches in antibiotic stewardship programs and optimizing prescription practices: a systematic review. *Artificial Intelligence in Medicine*. 2025;162:103089. <https://doi.org/10.1016/j.artmed.2025.103089>
7. European Centre for Disease Prevention and Control. *Antimicrobial resistance in the EU/EEA (EARS-Net): Annual Epidemiological Report for 2024*. Stockholm: ECDC; 2025. <https://www.ecdc.europa.eu/en/publications-data/antimicrobial-resistance-eueea-ears-net-annual-epidemiological-report-2024>
8. European Centre for Disease Prevention and Control. European Antimicrobial Resistance Surveillance Network (EARS-Net). <https://www.ecdc.europa.eu/en/about-us/networks/disease-networks-and-laboratory-networks/ears-net-data>
9. Tacconelli E, et al. Linking antimicrobial resistance surveillance to antibiotic policy in healthcare settings: the COMBACTE-Magnet EPI-Net COACH project. *Journal of Antimicrobial Chemotherapy*. 2020. <https://pmc.ncbi.nlm.nih.gov/articles/PMC7719409/>
10. Mo W, Qi Z, Liu Y. Learning optimal distributionally robust individualized treatment rules. *Journal of the American Statistical Association*. 2021. <https://pmc.ncbi.nlm.nih.gov/articles/PMC8221611/>
11. Wang Y, Liu A, Yang J, et al. Clinical knowledge-guided deep reinforcement learning for sepsis antibiotic dosing recommendations. *Artificial Intelligence in Medicine*. 2024;150:102811. <https://doi.org/10.1016/j.artmed.2024.102811>
