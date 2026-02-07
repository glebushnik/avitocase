# Шаг 6: NLP-маркеры стиля + обновленный ranking

## 1) Facts from data
- После очистки: 89,982 пользователей; sellers=49,547; eligible pool=39,441.
- Базовый blend: behavior 70% + style 30%.
- Сформирован `top-1000`: `analysis/step6_outputs/top1000_candidates_step6_text.csv`.

## 2) NLP/Style markers
### Professionalism
- Сигналы: quality/transaction keywords + seller rating + скорость ответа.
### Charisma
- Сигналы: CTA keywords + комментарии (объем/тональность/длина) + followers.
### Expertise
- Сигналы: domain keywords + questions_per_ad + deal_success_rate + текстовая специфичность (цифры/длина).

## 3) Ranking snapshot
- Top-1: user_id=baffd690cbc60a62, final_score=86.25, behavior_score=89.93, style_score=77.66.
- Overlap с step5 behavior-only top1000: 758 (75.80%), jaccard=0.6103.

## 4) Sensitivity (blend weight)
- style_20: overlap vs step5 = 848 (84.80%), jaccard=0.7361.
- style_30: overlap vs step5 = 758 (75.80%), jaccard=0.6103.
- style_40: overlap vs step5 = 670 (67.00%), jaccard=0.5038.

## 5) Interpretations (not facts)
- NLP-маркеры меняют состав top-листа умеренно, сохраняя большую часть поведенческого ядра.
- На synthetic-тексте словари работают как explainable layer, но для production-нужна валидация на реальных UGC-текстах.

## 6) ASSUMPTIONS
- ASSUMPTION: агрегированные поля комментариев (`comment_*`) можно использовать как прокси харизмы при отсутствии сырых комментариев.
- ASSUMPTION: словари ключевых фраз достаточны для MVP-ранжирования и презентации, хотя не заменяют full NLP-модель.

## Артефакты
- `analysis/step6_outputs/top1000_candidates_step6_text.csv`
- `analysis/step6_outputs/eligible_scores_step6.csv`
- `analysis/step6_outputs/blend_sensitivity.csv`
- `analysis/step6_outputs/marker_summary.csv`
- `analysis/step6_outputs/top1000_category_marker_scores.csv`
- `analysis/step6_outputs/top_terms_professionalism.csv`
- `analysis/step6_outputs/top_terms_charisma.csv`
- `analysis/step6_outputs/top_terms_expertise.csv`