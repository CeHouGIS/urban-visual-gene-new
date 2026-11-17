import copy
import json
from pathlib import Path
import pytest
from common import validate_config
from reuse_metrics import summarize_reuse

ROOT=Path(__file__).resolve().parents[2]


def config():return json.loads((ROOT/'configs/experiment.json').read_text())


def row(i,gene,delta,country='A'):
    return {'image_index':i,'gene_key':gene,'city_id':str(i),'country':country,
            'continent':'continent','baseline_mae':.02,'mae_increase':delta,'rgb_change_mae':.001}


def test_repeated_activation_is_not_functional_reuse():
    settings=config()['protocol']['reuse_evaluation']
    rows=[row(i,'dead',0) for i in range(4)]
    rows += [row(0,'useful',.001),row(1,'useful',.0008,'B'),row(2,'useful',-.001)]
    result=summarize_reuse(rows,['dead','useful','unused'],4,settings)
    assert result['genes_active_in_at_least_two_images']==2
    assert result['genes_with_repeated_measurable_benefit']==1
    assert result['genes_with_repeated_measurable_benefit_across_countries']==1
    genes={r['gene_key']:r for r in result['per_gene']}
    assert not genes['dead']['reused_with_measurable_benefit']
    assert genes['useful']['negative_mae_increase_city_count']==1
    assert genes['unused']['active_city_images']==0


def test_benefit_must_repeat_across_cities_and_country_breadth_is_separate():
    settings=config()['protocol']['reuse_evaluation']
    result=summarize_reuse([row(0,'a',.001),row(1,'a',.001),row(2,'b',.001)],['a','b'],3,settings)
    assert result['genes_with_repeated_measurable_benefit']==1
    assert result['genes_with_repeated_measurable_benefit_across_countries']==0
    with pytest.raises(ValueError):summarize_reuse([row(0,'a',.001)]*2,['a'],1,settings)


def test_reuse_priority_schedule_and_invalid_primary_rate():
    c=config();validate_config(c)
    schedule=c['model']['rate_schedule'];n=c['training']['max_updates']//len(schedule)
    assert {c['model']['budgets'][i]:schedule.count(i)*n for i in set(schedule)}=={8:1000,16:1000,32:2000,64:4000}
    assert c['protocol']['primary_gene_cap']==64
    bad=copy.deepcopy(c);bad['protocol']['primary_rate_index']=0
    with pytest.raises(ValueError):validate_config(bad)
    bad=copy.deepcopy(c);bad['model']['rate_schedule']=[9]
    with pytest.raises(ValueError):validate_config(bad)
