"""Stage 2 entrypoint: split -> climatology -> anomalies -> weekly means (task #3).

Cardinal rule: fit climatology + normalizer on TRAIN ONLY, then apply to val/test.
"""
# TODO: split_by_year -> fit_climatology(train) -> to_anomaly -> weekly_mean -> cache
if __name__ == "__main__":
    raise NotImplementedError
