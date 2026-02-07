# Avito Case — DS Analysis

Этот репозиторий содержит полный DS-пайплайн по кейсу `Авито Лента`: от аудита данных до выбора пилотного сегмента и формирования outreach short-list.

## Что за что отвечает

### Входные данные и постановка
- `07_02_Зимняя_школа_ВШБ_Марина_Кирякина_Авито.pdf`
  - Бизнес-постановка кейса и требования.
- `07_02_Зимняя_продуктовая_школа_кейс_Авито_описание_датасета.xlsx`
  - Описание полей датасета.
- `avito_lenta_dataset.csv`
  - Основной датасет (90k строк).
- `agents.md`
  - Правила DS-работы и формат финальных выводов.

### Аналитические шаги
- `analysis/step2_4_eda.py`
  - Шаги 2–4: data audit, EDA, baseline-сегментация Pro-users.
- `analysis/step5_scoring.py`
  - Шаг 5: поведенческий скоринг и top-1000 кандидатов.
- `analysis/step6_nlp.py`
  - Шаг 6: NLP/текстовые маркеры + обновленный ranking.
- `analysis/step7_pilot_strategy.py`
  - Шаг 7: выбор пилотного сегмента и fallback на основе концентрации/вовлеченности.

### Краткие отчеты по шагам
- `analysis/step2_4_summary.md`
- `analysis/step5_summary.md`
- `analysis/step6_summary.md`
- `analysis/step7_summary.md`

### Артефакты по шагам
- `analysis/step2_4_outputs/`
  - Пропуски, логические проверки, квантильные таблицы, baseline rules.
- `analysis/step5_outputs/`
  - `top1000_candidates.csv`, sensitivity по порогам и весам.
- `analysis/step6_outputs/`
  - `top1000_candidates_step6_text.csv`, blend sensitivity, NLP-маркеры и term-листы.
- `analysis/step7_outputs/`
  - Категориальные метрики и robustness, профиль рекомендованного сегмента.

### Финальные материалы для презентации и outreach
- `analysis/final_outputs/pilot_segment_config.json`
  - Зафиксированное решение: pilot=`Электроника`, fallback=`Авто`.
- `analysis/final_outputs/shortlist_top200_electronics_step6_with_tiers.csv`
  - Primary outreach shortlist (200 кандидатов).
- `analysis/final_outputs/shortlist_top200_auto_step6_with_tiers.csv`
  - Fallback shortlist по Авто (193 кандидата из текущего top-1000).
- `analysis/final_outputs/dataset_block_slides.md`
  - Готовый контент для 2 слайдов датасет-блока.
- `analysis/final_outputs/slide1_step6_case_categories.csv`
- `analysis/final_outputs/slide2_electronics_vs_auto_priority.csv`
- `analysis/final_outputs/slide2_electronics_profile_key_metrics.csv`

## Как воспроизвести результаты

Запускать из корня проекта:

```bash
python analysis/step2_4_eda.py
python analysis/step5_scoring.py
python analysis/step6_nlp.py
python analysis/step7_pilot_strategy.py
```

## Текущее финальное решение

- Пилотный сегмент: `Электроника`
- Fallback-сегмент: `Авто`
- Основной shortlist для запуска outreach:  
  `analysis/final_outputs/shortlist_top200_electronics_step6_with_tiers.csv`

