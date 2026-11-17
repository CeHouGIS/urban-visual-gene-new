"""CPU statistics distinguish repeated activation from useful cross-city reuse."""
import math


def summarize_reuse(records, genes, city_images, settings):
    """Each record deletes one active gene, with all weights and other codes fixed.

    Thresholds are descriptive sensitivity choices, not significance tests.
    Country breadth is reported separately from repeated-city usefulness.
    """
    if city_images < 1 or len(set(genes)) != len(genes):
        raise ValueError('Invalid image count or duplicate gene keys')
    grouped = {key: [] for key in genes}
    seen = set()
    for row in records:
        key = row['gene_key']
        pair = (row['image_index'], key)
        if key not in grouped or pair in seen or not 0 <= row['image_index'] < city_images:
            raise ValueError('Invalid or duplicated deletion record')
        if not all(math.isfinite(row[k]) for k in ['baseline_mae', 'mae_increase', 'rgb_change_mae']):
            raise ValueError('Nonfinite deletion measurement')
        if row['baseline_mae'] < 0 or row['rgb_change_mae'] < 0:
            raise ValueError('Negative absolute error')
        seen.add(pair)
        grouped[key].append(row)
    relative = settings['material_relative_mae_increase']
    absolute = settings['material_absolute_mae_increase']
    output = []
    for key, rows in grouped.items():
        useful = [r for r in rows if r['mae_increase'] > max(absolute, relative*r['baseline_mae'])]
        cities = {r['city_id'] for r in useful}
        countries = {r['country'] for r in useful}
        n = len(rows)
        output.append({'gene_key':key, 'active_city_images':n,
            'active_fraction':n/city_images, 'materially_helpful_city_count':len(cities),
            'materially_helpful_country_count':len(countries),
            'materially_helpful_continent_count':len({r['continent'] for r in useful}),
            'materially_helpful_fraction_when_active':len(useful)/n if n else 0.,
            'mean_deletion_mae_increase_when_active':sum(r['mae_increase'] for r in rows)/n if n else 0.,
            'mean_deletion_mae_increase_all_images':sum(r['mae_increase'] for r in rows)/city_images,
            'positive_mae_increase_city_count':len({r['city_id'] for r in rows if r['mae_increase']>0}),
            'negative_mae_increase_city_count':len({r['city_id'] for r in rows if r['mae_increase']<0}),
            'reused_with_measurable_benefit':len(cities)>=settings['minimum_helpful_cities'],
            'reused_with_measurable_benefit_across_countries':len(cities)>=settings['minimum_helpful_cities'] and len(countries)>=settings['minimum_helpful_countries']})
    output.sort(key=lambda r:(-r['materially_helpful_city_count'],-r['mean_deletion_mae_increase_all_images'],r['gene_key']))
    return {'city_images':city_images, 'dictionary_genes':len(genes),
        'active_gene_calls':len(records),
        'genes_active_in_at_least_two_images':sum(r['active_city_images']>=2 for r in output),
        'genes_with_repeated_measurable_benefit':sum(r['reused_with_measurable_benefit'] for r in output),
        'genes_with_repeated_measurable_benefit_across_countries':sum(r['reused_with_measurable_benefit_across_countries'] for r in output),
        'materially_helpful_gene_calls':sum(r['mae_increase']>max(absolute,relative*r['baseline_mae']) for r in records),
        'thresholds':settings, 'per_gene':output,
        'interpretation':'Fixed-code deletion measures conditional reconstruction contribution. Repeated activation alone is insufficient; this does not establish semantic identity or statistical significance.'}
