# Шаг 7: Выбор сегмента для пилота

## 1) Facts from data
- Общая база после очистки: 89,982 пользователей; sellers: 49,547.
- Strict сегмент: 193 (0.214% от всех; 0.390% от sellers).
- Relaxed сегмент: 881 (0.979% от всех; 1.778% от sellers).
- Финальный step6 top-1000: 1000 (1.111% от всех; 2.018% от sellers).

## 2) Категории из условия кейса (Авто/Недвижимость/Электроника/Хобби)
- Метрики для выбора: `concentration_idx` (доля в сегменте / доля у sellers), `engagement_idx` (сводный индекс вовлеченности), `priority_score = concentration_idx * engagement_idx`.

### Step6 top-1000 snapshot
- Электроника: users=226, concentration_idx=1.183, engagement_idx=1.191, priority_score=1.409.
- Авто: users=193, concentration_idx=1.205, engagement_idx=1.157, priority_score=1.394.
- Недвижимость: users=103, concentration_idx=0.941, engagement_idx=1.282, priority_score=1.206.
- Хобби и отдых: users=92, concentration_idx=0.771, engagement_idx=1.261, priority_score=0.972.

## 3) Recommendation
- Рекомендованный сегмент для первой волны пилота: **Про-пользователи категории 'Электроника'** (на базе step6 top-1000).
- Обоснование: в step6 `Электроника` имеет users=226, concentration_idx=1.183, engagement_idx=1.191, priority_score=1.409.
- Устойчивость: `Электроника` занимает 1-е место в 3 из 4 сценариев, средний ранг=1.25.

## 4) ASSUMPTIONS
- ASSUMPTION: `priority_score = concentration_idx * engagement_idx` является рабочим прокси пилотного потенциала (сетевой эффект + шанс на вовлечение).
- ASSUMPTION: step6 top-1000 — лучший operational proxy для 'первых блогеров' на текущем этапе, т.к. в нем учтены и поведенческие, и текстовые маркеры.

## Артефакты
- `analysis/step7_outputs/pro_user_share_by_definition.csv`
- `analysis/step7_outputs/category_metrics_all_scenarios.csv`
- `analysis/step7_outputs/case_category_metrics.csv`
- `analysis/step7_outputs/case_category_robustness.csv`
- `analysis/step7_outputs/recommended_segment_profile.csv`
- `analysis/step7_outputs/recommended_segment_marketing_mix.csv`