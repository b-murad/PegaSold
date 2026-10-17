"""Visualization helpers for the from-scratch DecisionTree.

Three ways to look at a fitted tree:
  * export_text(tree, ...)  -> plain-text tree (terminal / reports)
  * plot_tree(tree, ...)    -> matplotlib figure (PNG/PDF/SVG)
  * to_dot(tree, ...)       -> Graphviz DOT string (render with `dot -Tpng`)

Pass the training X, y to get per-node sample counts and impurities
(the tree itself only stores leaf information).
"""

import numpy as np


def _is_leaf(node):
    return node.feature_index is None


def _collect_stats(tree, X, y):
    """Route the data through the tree; return {id(node): (n, impurity, y_subset)}."""
    if X is None or y is None:
        return {}
    X = np.asarray(X)
    y = np.asarray(y)
    if tree.task == "regression":
        y = y.astype(float)
    stats = {}

    def walk(node, idx):
        ys = y[idx]
        imp = tree._impurity(ys) if len(ys) else 0.0
        stats[id(node)] = (len(idx), imp, ys)
        if not _is_leaf(node):
            mask = X[idx, node.feature_index] <= node.threshold
            walk(node.left, idx[mask])
            walk(node.right, idx[~mask])

    walk(tree.root, np.arange(len(y)))
    return stats


def _names(tree, feature_names):
    if feature_names is None:
        return [f"x[{i}]" for i in range(tree.n_features_in_)]
    if len(feature_names) != tree.n_features_in_:
        raise ValueError("feature_names length must equal number of features")
    return list(feature_names)


def _class_names(tree, class_names):
    if tree.task != "classification":
        return None
    if class_names is None:
        return [str(c) for c in tree.classes_]
    if len(class_names) != len(tree.classes_):
        raise ValueError("class_names length must equal number of classes")
    return list(class_names)


def _node_summary(tree, node, stats, cnames):
    """Return (n, impurity, counts_or_None, prediction_text)."""
    st = stats.get(id(node))
    if tree.task == "classification":
        if st is not None:
            ys = st[2]
            counts = [int(np.sum(ys == c)) for c in tree.classes_]
        elif _is_leaf(node):
            counts = [int(node.class_counts.get(c.item() if hasattr(c, "item") else c, 0))
                      for c in tree.classes_]
        else:
            return None, None, None, None
        pred = cnames[int(np.argmax(counts))] if sum(counts) else "?"
        return (st[0] if st else sum(counts)), (st[1] if st else None), counts, pred
    # regression
    if st is not None and st[0] > 0:
        return st[0], st[1], None, f"{np.mean(st[2]):.3f}"
    if _is_leaf(node):
        return None, None, None, f"{float(node.value):.3f}"
    return None, None, None, None


# ----------------------------------------------------------------- text ----
def export_text(tree, feature_names=None, X=None, y=None, decimals=3):
    """Return the tree as indented text, similar to sklearn.tree.export_text."""
    if tree.root is None:
        raise ValueError("fit must be called first")
    names = _names(tree, feature_names)
    cnames = _class_names(tree, None)
    stats = _collect_stats(tree, X, y)
    lines = []

    def leaf_text(node):
        n, _, counts, pred = _node_summary(tree, node, stats, cnames)
        if tree.task == "classification":
            label, val = "class", pred
        else:
            label, val = "value", f"{float(node.value):.{decimals}f}"
        extra = ""
        if n is not None:
            extra = f"  (samples={n}" + (f", counts={counts}" if counts else "") + ")"
        return f"{label}: {val}{extra}"

    def rec(node, depth):
        pad = "|   " * depth
        if _is_leaf(node):
            lines.append(f"{pad}|--- {leaf_text(node)}")
            return
        name = names[node.feature_index]
        thr = f"{node.threshold:.{decimals}f}"
        lines.append(f"{pad}|--- {name} <= {thr}")
        rec(node.left, depth + 1)
        lines.append(f"{pad}|--- {name} >  {thr}")
        rec(node.right, depth + 1)

    rec(tree.root, 0)
    return "\n".join(lines)


# ------------------------------------------------------------- graphviz ----
def to_dot(tree, feature_names=None, class_names=None, X=None, y=None, decimals=3):
    """Return a Graphviz DOT description. Render: dot -Tpng tree.dot -o tree.png"""
    if tree.root is None:
        raise ValueError("fit must be called first")
    names = _names(tree, feature_names)
    cnames = _class_names(tree, class_names)
    stats = _collect_stats(tree, X, y)
    lines = ["digraph Tree {", '  node [shape=box, style="rounded,filled", fontname="Helvetica"];']
    counter = [0]

    def rec(node):
        my = counter[0]
        counter[0] += 1
        n, imp, counts, pred = _node_summary(tree, node, stats, cnames)
        parts = []
        if not _is_leaf(node):
            parts.append(f"{names[node.feature_index]} <= {node.threshold:.{decimals}f}")
        if imp is not None:
            parts.append(f"{tree.criterion} = {imp:.{decimals}f}")
        if n is not None:
            parts.append(f"samples = {n}")
        if counts is not None:
            parts.append(f"counts = {counts}")
        if pred is not None:
            parts.append(("class = " if tree.task == "classification" else "value = ") + str(pred))
        fill = "#ffffff" if not _is_leaf(node) else "#d9f2d0"
        label = "\\n".join(parts)
        lines.append(f'  {my} [label="{label}", fillcolor="{fill}"];')
        if not _is_leaf(node):
            left = rec(node.left)
            right = rec(node.right)
            lines.append(f'  {my} -> {left} [label="True"];')
            lines.append(f'  {my} -> {right} [label="False"];')
        return my

    rec(tree.root)
    lines.append("}")
    return "\n".join(lines)


# ----------------------------------------------------------- matplotlib ----
def _layout(root):
    pos = {}
    counter = [0]

    def rec(node, depth):
        if _is_leaf(node):
            x = float(counter[0])
            counter[0] += 1
        else:
            x = (rec(node.left, depth + 1) + rec(node.right, depth + 1)) / 2
        pos[id(node)] = (x, -depth)
        return x

    rec(root, 0)
    return pos, counter[0]


def _tree_depth(node):
    return 0 if _is_leaf(node) else 1 + max(_tree_depth(node.left), _tree_depth(node.right))


def plot_tree(tree, feature_names=None, class_names=None, X=None, y=None,
              ax=None, figsize=None, fontsize=8, decimals=3, save_path=None):
    """Draw the tree with matplotlib. Returns the Axes. Nodes are colored by
    majority class (classification) or by predicted value (regression)."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import colors

    if tree.root is None:
        raise ValueError("fit must be called first")
    names = _names(tree, feature_names)
    cnames = _class_names(tree, class_names)
    stats = _collect_stats(tree, X, y)
    pos, n_leaves = _layout(tree.root)
    depth = _tree_depth(tree.root)

    if ax is None:
        figsize = figsize or (max(6, 2.0 * n_leaves), max(4, 1.8 * (depth + 1)))
        _, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    # color scales
    if tree.task == "classification":
        palette = matplotlib.colormaps["tab10"]
    else:
        vals = []

        def gather(n):
            if _is_leaf(n):
                vals.append(float(n.value))
            else:
                gather(n.left); gather(n.right)

        gather(tree.root)
        lo, hi = min(vals), max(vals)
        norm = colors.Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1)

    def node_color(node):
        n, imp, counts, pred = _node_summary(tree, node, stats, cnames)
        if tree.task == "classification":
            if counts and sum(counts):
                k = int(np.argmax(counts))
                purity = max(counts) / sum(counts)
                base = np.array(colors.to_rgb(palette(k % 10)))
                return tuple(1 - (1 - base) * (0.25 + 0.75 * purity))
            return (1, 1, 1)
        v = float(pred) if pred is not None else 0.0
        base = np.array(matplotlib.colormaps["viridis"](norm(v))[:3])
        return tuple(1 - (1 - base) * 0.45)

    def draw(node):
        x, yv = pos[id(node)]
        if not _is_leaf(node):
            for child, lab in ((node.left, "True"), (node.right, "False")):
                cx, cy = pos[id(child)]
                ax.plot([x, cx], [yv, cy], color="gray", lw=1, zorder=1)
                ax.text((x + cx) / 2, (yv + cy) / 2, lab, fontsize=fontsize - 1,
                        color="dimgray", ha="center", va="center",
                        bbox=dict(fc="white", ec="none", pad=0.5))
                draw(child)
        n, imp, counts, pred = _node_summary(tree, node, stats, cnames)
        parts = []
        if not _is_leaf(node):
            parts.append(f"{names[node.feature_index]} <= {node.threshold:.{decimals}g}")
        if imp is not None:
            parts.append(f"{tree.criterion} = {imp:.{decimals}f}")
        if n is not None:
            parts.append(f"samples = {n}")
        if counts is not None:
            parts.append(f"counts = {counts}")
        if pred is not None:
            parts.append(("class = " if tree.task == "classification" else "value = ") + str(pred))
        ax.text(x, yv, "\n".join(parts), ha="center", va="center", fontsize=fontsize,
                zorder=2, bbox=dict(boxstyle="round,pad=0.4", fc=node_color(node), ec="black", lw=0.8))

    draw(tree.root)
    ax.set_xlim(-0.8, max(n_leaves - 1, 0) + 0.8)
    ax.set_ylim(-depth - 0.6, 0.6)
    if save_path:
        ax.figure.savefig(save_path, dpi=200, bbox_inches="tight")
    return ax


def render_dot(dot_string, out_path="tree.png"):
    """Render a DOT string with the `dot` binary (Graphviz must be installed)."""
    import shutil, subprocess
    if shutil.which("dot") is None:
        raise RuntimeError("Graphviz 'dot' not found; install it (e.g. sudo apt install graphviz)")
    fmt = out_path.rsplit(".", 1)[-1]
    subprocess.run(["dot", f"-T{fmt}", "-o", out_path], input=dot_string.encode(), check=True)
    return out_path
