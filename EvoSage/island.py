class AdaptiveGridArchive:
  """Archive that bins solutions into a grid that adapts to score ranges."""

  def __init__(self, dim: int, bins: int = 10, metric_names=None):
    self.dim = dim
    self.bins = [bins] * dim
    self.bounds_min = [float("inf")] * dim
    self.bounds_max = [float("-inf")] * dim
    self.metric_names = list(metric_names or [f"obj{i}" for i in range(dim)])
    self.archive: dict[tuple[int, ...], dict[str, object]] = {}

  def _update_bounds(self, point: tuple[float, ...]) -> None:
    for i, v in enumerate(point):
      if v < self.bounds_min[i]:
        self.bounds_min[i] = v
      if v > self.bounds_max[i]:
        self.bounds_max[i] = v

  def _get_cell(self, point: tuple[float, ...]) -> tuple[int, ...]:
    idx = []
    for i, v in enumerate(point):
      span = self.bounds_max[i] - self.bounds_min[i]
      if span == 0:
        b = 0
      else:
        b = int((v - self.bounds_min[i]) / span * self.bins[i])
      idx.append(max(0, min(self.bins[i] - 1, b)))
    return tuple(idx)

  def add(self, seq: str, score: tuple[float, ...], gen: int) -> None:
    self._update_bounds(score)
    cell = self._get_cell(score)
    cur = self.archive.get(cell)
    if cur is None or score[0] > cur["score"][0]:
      self.archive[cell] = {"seq": seq, "score": score, "gen": gen}

  def values(self):
    rows = []
    for entry in self.archive.values():
      row = {"seq": entry["seq"], "gen": entry.get("gen")}
      for name, val in zip(self.metric_names, entry["score"]):
        row[name] = val
      rows.append(row)
    return rows


def cluster_niching(df, elite, n_clusters: int = 5, per_cluster: int = 5):
  """Cluster elite sequences and keep top candidates from each cluster."""
  if not elite:
    return []
  seqs = [cand["seq"] for cand in elite]
  mapping = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
  X = [[mapping[aa] for aa in seq] for seq in seqs]
  from sklearn.cluster import KMeans

  n_clusters = max(1, min(n_clusters, len(seqs)))
  labels = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0).fit_predict(X)
  selected = []
  for lab in range(n_clusters):
    idxs = [i for i, lbl in enumerate(labels) if lbl == lab]
    subset = df[df["seq"].isin([seqs[i] for i in idxs])]
    top = subset.sort_values("additive", ascending=False).head(per_cluster)
    for seq in top["seq"]:
      cand = next(c for c in elite if c["seq"] == seq)
      selected.append(cand)
  return selected
