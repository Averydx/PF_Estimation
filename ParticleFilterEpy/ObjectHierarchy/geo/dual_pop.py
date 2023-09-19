from epymorph.geo import Geo,CentroidDType
import numpy as np


def load() -> Geo:
    label = ['AZ','CA']
    return Geo(
        nodes=len(label),
        labels=label,
        data={
            'label': np.array(label, dtype=str),
            'geoid': np.array(['04','05'], dtype=str),
            'centroid': np.array([(-111.856111, 34.566667)], dtype=CentroidDType),
            'population': np.array([10_000_000,5_000_000], dtype=np.int64),
            'commuters': np.array([[0,10000],
                                   [10000,0]], dtype=np.int64)
        }
    )
