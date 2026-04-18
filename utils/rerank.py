import math

from rdkit import Chem

from utils.chemistry_parse import canonical_smiles


def _canonicalize_prediction(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return canonical_smiles(Chem.MolToSmiles(mol))


def _sort_scores(score_map):
    ranked = sorted(score_map.items(), key=lambda x: -x[1])
    return [x[0] for x in ranked], [x[1] for x in ranked]


def _merge_prediction_group(answer_group, prob_group, keep_invalid=False):
    merged = {}
    fallback = {}
    for answers, probs in zip(answer_group, prob_group):
        local_prob = {}
        for smi, score in zip(answers, probs):
            fallback[smi] = fallback.get(smi, 0.0) + math.exp(score)
            key = _canonicalize_prediction(smi)
            if key is None:
                if not keep_invalid:
                    continue
                key = smi
            local_prob[key] = local_prob.get(key, 0.0) + math.exp(score)

        for smi, score in local_prob.items():
            merged[smi] = merged.get(smi, 0.0) + score

    if len(merged) == 0:
        merged = fallback

    merged_logits = {
        smi: math.log(score + 1e-30) for smi, score in merged.items()
    }
    return _sort_scores(merged_logits)


def rerank_ensemble(
    answer_group,
    prob_group,
    keep_invalid=False,
    score_alpha=1.0,
):
    if score_alpha <= 0:
        raise ValueError('score_alpha should be > 0 for ensemble ranking')

    rank_scores = {}
    for answers, probs in zip(answer_group, prob_group):
        preds, _ = _merge_prediction_group(
            [answers], [probs], keep_invalid=keep_invalid
        )
        for rank_idx, smi in enumerate(preds):
            rank_scores[smi] = rank_scores.get(smi, 0.0) + (
                1.0 / (score_alpha * rank_idx + 1.0)
            )

    return _sort_scores(rank_scores)


def rerank_predictions(
    answer_group,
    prob_group,
    rank_compute='log_logits',
    keep_invalid=False,
    score_alpha=0.1,
):
    if rank_compute == 'log_logits':
        return _merge_prediction_group(
            answer_group, prob_group, keep_invalid=keep_invalid
        )
    if rank_compute == 'ensemble':
        return rerank_ensemble(
            answer_group,
            prob_group,
            keep_invalid=keep_invalid,
            score_alpha=score_alpha,
        )
    raise ValueError(f'Unsupported rank_compute: {rank_compute}')
