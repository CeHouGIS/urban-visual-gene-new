"""Capacity-aware equal allocation at each geographic level."""
from collections import defaultdict
import random


def balanced_allocation(capacities,total,seed):
    capacities={key:int(value) for key,value in capacities.items() if value>0}
    if total<0 or sum(capacities.values())<total:
        raise ValueError(f'Insufficient capacity: need {total}, have {sum(capacities.values())}')
    order=sorted(capacities,key=repr)
    random.Random(str(seed)).shuffle(order)
    result={key:0 for key in order}
    remaining=total
    while remaining:
        active=[key for key in order if result[key]<capacities[key]]
        share=max(1,remaining//len(active))
        for key in active:
            take=min(share,capacities[key]-result[key],remaining)
            result[key]+=take
            remaining-=take
            if not remaining:
                break
    return result


def hierarchical_allocation(capacities,total,seed):
    """Keys are (continent,country,GHSL city ID), values are usable grid counts."""
    continents=defaultdict(int)
    countries=defaultdict(int)
    for (continent,country,city),capacity in capacities.items():
        continents[continent]+=capacity
        countries[continent,country]+=capacity
    continent_quota=balanced_allocation(continents,total,f'{seed}/continents')
    result={}
    for continent,quota in continent_quota.items():
        country_caps={country:value for (region,country),value in countries.items() if region==continent}
        country_quota=balanced_allocation(country_caps,quota,f'{seed}/{continent}/countries')
        for country,number in country_quota.items():
            city_caps={city:value for (region,nation,city),value in capacities.items() if (region,nation)==(continent,country)}
            city_quota=balanced_allocation(city_caps,number,f'{seed}/{continent}/{country}/cities')
            result.update({(continent,country,city):value for city,value in city_quota.items()})
    return result
