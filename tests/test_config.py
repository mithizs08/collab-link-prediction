import random

from linkpred.config import load_config, set_seed


def test_default_config_has_required_sections():
    cfg = load_config("configs/default.yaml")
    for key in ("seed", "seeds", "paths", "data", "splits", "negatives", "models"):
        assert key in cfg
    assert cfg["seeds"] == [0, 1, 2, 3, 4]
    assert cfg["data"]["max_authors_per_work"] == 25
    assert set(cfg["splits"]) >= {"train", "val", "test"}


def test_split_cutoffs_are_ordered_when_set():
    # Guards against a val/test window that overlaps or precedes training.
    cfg = load_config("configs/default.yaml")
    s = cfg["splits"]
    cutoffs = [s[p]["cutoff"] for p in ("train", "val", "test")]
    if all(c is not None for c in cutoffs):
        assert cutoffs == sorted(cutoffs) and len(set(cutoffs)) == 3
        for p in ("train", "val", "test"):
            assert s[p]["window_end"] >= s[p]["cutoff"]
        # A label window must end before the next period's cutoff, otherwise the
        # same new edges would be labels in one period and graph input in the next.
        assert s["train"]["window_end"] < s["val"]["cutoff"]
        assert s["val"]["window_end"] < s["test"]["cutoff"]


def test_set_seed_is_reproducible():
    set_seed(123)
    a = [random.random() for _ in range(5)]
    set_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_decided_splits_and_target():
    cfg = load_config("configs/default.yaml")
    s = cfg["splits"]
    assert [(s[p]["cutoff"], s[p]["window_end"]) for p in ("train", "val", "test")] == [
        (2023, 2023), (2024, 2024), (2025, 2025)]
    assert set(cfg["data"]["exclude_types"]) == {"erratum", "peer-review", "paratext"}


# ---------------------------------------------------------------- config access paths (audit m5 / N5)
# A static (AST) analysis that resolves every config access to its FULL key path, e.g.
#   lcfg = cfg["models"]["lightgbm"]; lcfg["search"]["num_leaves"]   ->  models.lightgbm.search.num_leaves
# so a key only counts as read when code reaches it through the config object, not when its bare name
# appears somewhere (`kv["lr"]` on a parsed string does not read models.node2vec.lr).
# Bindings: `cfg` / `config` / `self.cfg` = the root (load_config result); `x = <config expr>` and
# `self.x = <config expr>`; function parameters receive the paths of the arguments passed at call sites
# anywhere in the project (iterated to a fixed point); `for k in ("a", "b")` makes k a known key set; a
# non-constant key is a wildcard ("*"); `{**cfg_x, ...}` keeps cfg_x's path. Bindings are unions, so a
# parameter or attribute bound to two paths reads both (over-approximation per access path, never per bare name).

import ast as _ast

ROOT_NAMES = {"cfg", "config"}
WILD = "*"
DICT_METHODS = {"get", "items", "keys", "values", "update", "pop", "setdefault", "copy"}


class _Analyzer:
    def __init__(self, files, cfg):
        self.cfg = cfg
        self.trees = {f: _ast.parse(f.read_text()) for f in files}
        self.funcs = {}                       # function name -> list of parameter-name lists
        for t in self.trees.values():
            for n in _ast.walk(t):
                if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                    self.funcs.setdefault(n.name, []).append([a.arg for a in n.args.args + n.args.kwonlyargs])
        self.param_paths = {}                 # (function name, parameter) -> set of paths
        self.accesses = set()

    def run(self, rounds=6):
        for _ in range(rounds):
            before = {k: set(v) for k, v in self.param_paths.items()}
            self.accesses = set()
            for t in self.trees.values():
                self.self_env = {}
                for _ in range(2):            # second pass sees self.x bound later in the file
                    self._scope(t.body, {}, None)
            if before == self.param_paths:
                break
        return self.accesses

    # -- resolution: returns a set of ("path", tuple) / ("str", value)
    def _resolve(self, e, env):
        if isinstance(e, _ast.Name):
            out = set(env.get(e.id, set()))
            if e.id in ROOT_NAMES:
                out.add(("path", ()))
            return out
        if isinstance(e, _ast.Constant) and isinstance(e.value, str):
            return {("str", e.value)}
        if isinstance(e, _ast.Attribute) and isinstance(e.value, _ast.Name) and e.value.id == "self":
            out = set(self.self_env.get(e.attr, set()))
            if e.attr in ROOT_NAMES:
                out.add(("path", ()))
            return out
        if isinstance(e, _ast.Subscript):
            return self._child(self._resolve(e.value, env), self._keys(e.slice, env))
        if (isinstance(e, _ast.Call) and isinstance(e.func, _ast.Attribute) and e.func.attr == "get" and e.args):
            return self._child(self._resolve(e.func.value, env), self._keys(e.args[0], env))
        if isinstance(e, _ast.Dict):
            return {x for k, v in zip(e.keys, e.values) if k is None for x in self._resolve(v, env) if x[0] == "path"}
        if isinstance(e, _ast.BoolOp):
            return {x for v in e.values for x in self._resolve(v, env)}
        if isinstance(e, _ast.IfExp):
            return self._resolve(e.body, env) | self._resolve(e.orelse, env)
        return set()

    def _node(self, p):
        node = self.cfg
        for k in p:
            if not isinstance(node, dict):
                return None
            if k == WILD:                         # any child: the union of the dict children's keys
                kids = [v for v in node.values() if isinstance(v, dict)]
                node = {kk: vv for kid in kids for kk, vv in kid.items()} if kids else 0
            else:
                node = node.get(k)
        return node

    def _keys(self, e, env):
        ks = {v for kind, v in self._resolve(e, env) if kind == "str"}
        return ks or {WILD}

    def _child(self, bases, keys):
        out = set()
        for kind, p in bases:
            if kind == "path":
                node = self._node(p)
                if not isinstance(node, dict) or p[-1:] in (("sage",), ("gat",)):
                    continue                      # below a leaf: not a config access
                for k in keys:
                    if k == WILD and not p:       # cfg[<unknown>]: not a resolvable config access
                        continue
                    if k != WILD and k not in node:
                        continue                  # not a config key (e.g. a column of a frame with the same name)
                    out.add(("path", p + (k,)))
                    self.accesses.add(p + (k,))
        return out

    def _bind(self, target, val, env):
        if isinstance(target, _ast.Name):
            env[target.id] = val                  # rebinding replaces (the name no longer holds the old value)
        elif isinstance(target, _ast.Attribute) and isinstance(target.value, _ast.Name) and target.value.id == "self":
            if val:
                self.self_env[target.attr] = self.self_env.get(target.attr, set()) | val

    def _scope(self, body, env, params):
        env = dict(env)
        if params:
            for name, ps in params:
                env[name] = env.get(name, set()) | ps
        for stmt in body:
            self._stmt(stmt, env)

    def _stmt(self, s, env):
        if isinstance(s, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            names = [a.arg for a in s.args.args + s.args.kwonlyargs]
            self._scope(s.body, env, [(n, self.param_paths.get((s.name, n), set())) for n in names])
            return
        if isinstance(s, _ast.ClassDef):
            self._scope(s.body, env, None)
            return
        if isinstance(s, (_ast.Assign, _ast.AnnAssign)) and s.value is not None:
            val = self._visit_expr(s.value, env)
            for t in (s.targets if isinstance(s, _ast.Assign) else [s.target]):
                if isinstance(t, _ast.Tuple) and isinstance(s.value, _ast.Tuple) and len(t.elts) == len(s.value.elts):
                    for tt, vv in zip(t.elts, s.value.elts):          # a, b = cfg["x"], cfg["y"]
                        self._bind(tt, self._resolve(vv, env), env)
                else:
                    self._bind(t, val, env)
            return
        if isinstance(s, (_ast.For, _ast.AsyncFor)):
            self._visit_expr(s.iter, env)
            it = s.iter
            if isinstance(it, (_ast.Tuple, _ast.List)) and all(isinstance(x, _ast.Constant) for x in it.elts):
                self._bind(s.target, {("str", x.value) for x in it.elts if isinstance(x.value, str)}, env)
            elif (isinstance(it, _ast.Call) and isinstance(it.func, _ast.Attribute) and it.func.attr == "items"
                  and isinstance(s.target, _ast.Tuple) and len(s.target.elts) == 2):
                self._bind(s.target.elts[1], self._child(self._resolve(it.func.value, env), {WILD}), env)
            for b in s.body + s.orelse:
                self._stmt(b, env)
            return
        for field in ("body", "orelse", "finalbody", "handlers"):
            for b in getattr(s, field, []) or []:
                if isinstance(b, _ast.ExceptHandler):
                    for bb in b.body:
                        self._stmt(bb, env)
                elif isinstance(b, _ast.stmt):
                    self._stmt(b, env)
        for name, value in _ast.iter_fields(s):
            if name in ("body", "orelse", "finalbody", "handlers"):
                continue
            for x in (value if isinstance(value, list) else [value]):
                if isinstance(x, _ast.expr):
                    self._visit_expr(x, env)
                elif isinstance(x, _ast.withitem):
                    self._visit_expr(x.context_expr, env)

    def _visit_expr(self, e, env):
        """Resolve e (recording accesses) and propagate argument paths into project functions' parameters."""
        for n in _ast.walk(e):
            if isinstance(n, _ast.comprehension) and isinstance(n.iter, (_ast.Tuple, _ast.List)):
                self._bind(n.target, {("str", x.value) for x in n.iter.elts
                                      if isinstance(x, _ast.Constant) and isinstance(x.value, str)}, env)
        for n in _ast.walk(e):
            if isinstance(n, (_ast.Subscript, _ast.Call)):
                self._resolve(n, env)
            if isinstance(n, _ast.Call):
                fname = n.func.id if isinstance(n.func, _ast.Name) else (n.func.attr if isinstance(n.func, _ast.Attribute) else None)
                if fname in DICT_METHODS:
                    continue
                for params in self.funcs.get(fname, []):
                    ps = params[1:] if params and params[0] == "self" and isinstance(n.func, _ast.Attribute) else params
                    if fname == "__init__":
                        continue
                    for i, a in enumerate(n.args):
                        if i < len(ps):
                            v = {x for x in self._resolve(a, env) if x[0] == "path"}
                            if v:
                                self.param_paths.setdefault((fname, ps[i]), set()).update(v)
                    for kw in n.keywords:
                        if kw.arg in params:
                            v = {x for x in self._resolve(kw.value, env) if x[0] == "path"}
                            if v:
                                self.param_paths.setdefault((fname, kw.arg), set()).update(v)
                # Experiment(cfg) -> __init__(self, cfg)
                if isinstance(n.func, _ast.Name):
                    for params in self.funcs.get("__init__", []):
                        for i, a in enumerate(n.args):
                            if i + 1 < len(params):
                                v = {x for x in self._resolve(a, env) if x[0] == "path"}
                                if v:
                                    self.param_paths.setdefault(("__init__", params[i + 1]), set()).update(v)
        return self._resolve(e, env)


def config_accesses(files, cfg):
    return _Analyzer(files, cfg).run()


def _leaf_paths(d, path=()):
    for k, v in d.items():
        if isinstance(v, dict) and k not in ("sage", "gat"):     # GNN grids are read as whole dicts
            yield from _leaf_paths(v, path + (k,))
        else:
            yield path + (k,)


def _is_read(leaf, accesses):
    return any(len(a) == len(leaf) and all(x == WILD or x == y for x, y in zip(a, leaf)) for a in accesses)


def test_config_access_analysis_resolves_paths(tmp_path):
    """The analysis follows aliases, parameters and loops, and ignores bare key names."""
    f = tmp_path / "m.py"
    f.write_text(
        "def helper(c):\n    return c['lr']\n"
        "def main(cfg):\n"
        "    n = cfg['models']['node2vec']\n    helper(n)\n"
        "    for k in ('a', 'b'):\n        cfg['grid'][k]\n"
        "    cfg['splits'][name]['cutoff']\n"
        "    kv = dict(x.split('=') for x in s)\n    kv['epochs']\n")
    acc = config_accesses([f], {"models": {"node2vec": {"lr": 1, "epochs": 1}}, "lr": 1,
                                  "grid": {"a": 1, "b": 1, "c": 1}, "splits": {"val": {"cutoff": 1}}})
    assert _is_read(("models", "node2vec", "lr"), acc)
    assert _is_read(("grid", "a"), acc) and _is_read(("grid", "b"), acc) and not _is_read(("grid", "c"), acc)
    assert _is_read(("splits", "val", "cutoff"), acc)
    assert not _is_read(("models", "node2vec", "epochs"), acc)     # kv['epochs'] is not a config access
    assert not _is_read(("lr",), acc)                               # only reached through models.node2vec


def test_every_config_key_is_read_by_code():
    """Audit m5 / N5: no decorative config keys. Every leaf key must be reached through its full config path."""
    import pathlib
    cfg = load_config("configs/default.yaml")
    files = list(pathlib.Path("src").rglob("*.py")) + list(pathlib.Path("scripts").glob("*.py"))
    acc = config_accesses(files, cfg)
    unused = [".".join(map(str, p)) for p in _leaf_paths(cfg) if not _is_read(p, acc)]
    assert not unused, f"config keys never read through their config path: {unused}"
