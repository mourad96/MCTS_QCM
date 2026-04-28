"""Rendering helpers: rich-tree printing, JSON dump, Markdown export, canvas and HTML generation."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from mcts_qcm.node import Node
from mcts_qcm.scoring import greedy_best_path


def _truncate(text: str, n: int = 110) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _node_label(node: Node) -> Text:
    """Format a single node line for the rich tree."""
    label = Text()
    if node.parent is None:
        label.append("ROOT ", style="bold")
        label.append(_truncate(node.idea, 140))
        return label

    audit = node.audit
    if audit is not None:
        passed = audit.num_passed
        color = {0: "red", 1: "red", 2: "yellow", 3: "green", 4: "bright_green"}[passed]
        label.append(f"[{audit.summary()}] ", style=f"bold {color}")
    else:
        label.append("[unaudited] ", style="dim")

    if node.dead:
        label.append("(dead) ", style="bold red")

    label.append(f"v={node.visits} q={node.mean_value:.2f}  ", style="cyan")
    label.append(_truncate(node.idea, 130))
    return label


def render_tree(root: Node) -> Tree:
    """Build a `rich.tree.Tree` from the MCTS tree rooted at ``root``."""
    tree = Tree(_node_label(root))

    def _add(parent_widget: Tree, n: Node) -> None:
        for child in n.children:
            sub = parent_widget.add(_node_label(child))
            _add(sub, child)

    _add(tree, root)
    return tree


def print_summary(root: Node, console: Console | None = None) -> None:
    """Print the tree, the greedy best path, and a small stats block."""
    console = console or Console()
    console.print(Panel.fit("MCTS QCM Search Tree", style="bold magenta"))
    console.print(render_tree(root))

    path = greedy_best_path(root)
    console.print()
    console.print(Panel.fit("Greedy best path (root → leaf)", style="bold green"))
    for i, node in enumerate(path):
        prefix = "ROOT" if i == 0 else f"  {i}."
        audit_str = f" [{node.audit.summary()}]" if node.audit is not None else ""
        console.print(f"{prefix}{audit_str} {_truncate(node.idea, 200)}")

    descendants = root.iter_descendants()
    audited = [n for n in descendants if n.audit is not None]
    pruned = [n for n in descendants if n.dead]
    console.print()
    console.print(
        f"[bold]Stats:[/bold] {len(descendants)} nodes  •  "
        f"{len(audited)} audited  •  {len(pruned)} pruned  •  "
        f"root visits={root.visits}"
    )


def to_json(root: Node) -> str:
    """JSON dump of the entire tree (pretty-printed)."""
    return json.dumps(root.to_dict(), indent=2, ensure_ascii=False)


def write_json(root: Node, path: str | Path) -> Path:
    p = Path(path)
    p.write_text(to_json(root), encoding="utf-8")
    return p


def to_markdown(root: Node) -> str:
    """Markdown export of the tree (for sharing in PRs / docs)."""
    lines: list[str] = ["# MCTS QCM Search Tree", ""]

    def _walk(n: Node, depth: int) -> None:
        indent = "  " * depth
        if n.parent is None:
            lines.append(f"{indent}- **ROOT**: {n.idea}")
        else:
            audit_str = ""
            if n.audit is not None:
                audit_str = f" [{n.audit.summary()}]"
            dead_str = " _(pruned)_" if n.dead else ""
            lines.append(
                f"{indent}- {audit_str} v={n.visits} q={n.mean_value:.2f}{dead_str} — {n.idea}"
            )
        for child in n.children:
            _walk(child, depth + 1)

    _walk(root, 0)

    lines.append("")
    lines.append("## Greedy best path")
    for i, node in enumerate(greedy_best_path(root)):
        if i == 0:
            lines.append(f"- **ROOT**: {node.idea}")
        else:
            audit_str = f" [{node.audit.summary()}]" if node.audit is not None else ""
            lines.append(f"  {i}.{audit_str} {node.idea}")

    return "\n".join(lines)


# ── Canvas generation ─────────────────────────────────────────────────────────

# Raw TypeScript template for the canvas.  Insertion markers:
#   __TREE_JSON_B64__       base-64 encoded JSON of the full tree dict
#   __BEST_PATH_IDS__       comma-separated list of node IDs on the best path
#   __DEFAULT_SELECTED_ID__ integer ID of the best-path leaf (default selection)
_CANVAS_TEMPLATE = r"""import {
  computeDAGLayout,
  useHostTheme,
  useCanvasState,
  Stack,
  Row,
  H1,
  H3,
  Grid,
  Stat,
  Divider,
  Callout,
  Table,
  Text,
} from "cursor/canvas";

// ── Types ────────────────────────────────────────────────────────────────────
type AuditResult = {
  novelty: boolean;
  resource: boolean;
  feasibility: boolean;
  alignment: boolean;
  reasons: Record<string, string>;
};
type TreeNode = {
  id: number;
  idea: string;
  depth: number;
  visits: number;
  value_sum: number;
  mean_value: number;
  dead: boolean;
  audit: AuditResult | null;
  children: TreeNode[];
};

// ── Injected data ─────────────────────────────────────────────────────────────
const TREE_DATA: TreeNode = JSON.parse(atob("__TREE_JSON_B64__"));
const BEST_PATH_IDS = new Set<number>([__BEST_PATH_IDS__]);
const DEFAULT_SELECTED_ID: number = __DEFAULT_SELECTED_ID__;

// ── Layout constants ──────────────────────────────────────────────────────────
const NW = 240;
const NH = 100;

// ── Flatten tree once at module level ─────────────────────────────────────────
const nodesMap = new Map<number, TreeNode>();
const edgeList: { from: string; to: string }[] = [];
(function flatten(node: TreeNode): void {
  nodesMap.set(node.id, node);
  for (const child of node.children) {
    edgeList.push({ from: String(node.id), to: String(child.id) });
    flatten(child);
  }
})(TREE_DATA);

const layout = computeDAGLayout({
  nodes: Array.from(nodesMap.keys()).map(id => ({ id: String(id) })),
  edges: edgeList,
  direction: "vertical",
  nodeWidth: NW,
  nodeHeight: NH,
  rankGap: 90,
  nodeGap: 24,
  padding: 32,
});

// ── Helpers ───────────────────────────────────────────────────────────────────
function trunc(text: string, n: number): string {
  const s = text.replace(/\s+/g, " ").trim();
  return s.length <= n ? s : s.slice(0, n - 1) + "\u2026";
}

const CHECKS = ["novelty", "resource", "feasibility", "alignment"] as const;
type CheckKey = typeof CHECKS[number];

const CHECK_LABELS: Record<CheckKey, string> = {
  novelty: "Novelty",
  resource: "Resource",
  feasibility: "Feasibility",
  alignment: "Alignment",
};

function auditPassed(audit: AuditResult, k: CheckKey): boolean {
  return { novelty: audit.novelty, resource: audit.resource, feasibility: audit.feasibility, alignment: audit.alignment }[k];
}

// ── NodeDetail ────────────────────────────────────────────────────────────────
function NodeDetail({ node, theme }: { node: TreeNode; theme: ReturnType<typeof useHostTheme> }) {
  const isRoot = node.depth === 0;
  return (
    <Stack gap={14}>
      <Grid columns={2} gap={10}>
        <Stat value={String(node.visits)} label="Visits" />
        <Stat
          value={`${Math.round(node.mean_value * 100)}%`}
          label="Mean score"
          tone={node.mean_value >= 0.75 ? "success" : node.mean_value < 0.5 ? "danger" : undefined}
        />
      </Grid>

      {node.dead && (
        <Callout tone="warning" title="Pruned">
          This node failed a hard-threshold check (Resource or Novelty) and was removed from further expansion.
        </Callout>
      )}

      {BEST_PATH_IDS.has(node.id) && !isRoot && (
        <Callout tone="success" title="On the best path">
          This node lies on the greedy best path from root to leaf.
        </Callout>
      )}

      <H3>Idea</H3>
      <Text size="small" style={{ lineHeight: 1.6 }}>{node.idea}</Text>

      {node.audit ? (
        <>
          <Divider />
          <H3>QCM Audit</H3>
          <Table
            headers={["Check", "Result", "Reason"]}
            rows={CHECKS.map(k => [
              CHECK_LABELS[k],
              auditPassed(node.audit!, k) ? "PASS" : "FAIL",
              node.audit!.reasons[k] ?? "\u2014",
            ])}
            rowTone={CHECKS.map(k => (auditPassed(node.audit!, k) ? "success" : "danger"))}
          />
        </>
      ) : (
        <Text tone="tertiary" size="small" italic>Root node — no QCM audit (this is the problem statement).</Text>
      )}

      <Divider />
      <Grid columns={3} gap={8}>
        <div>
          <Text size="small" tone="secondary">Depth</Text>
          <Text size="small" weight="semibold">{node.depth}</Text>
        </div>
        <div>
          <Text size="small" tone="secondary">Value sum</Text>
          <Text size="small" weight="semibold">{node.value_sum.toFixed(2)}</Text>
        </div>
        <div>
          <Text size="small" tone="secondary">Node ID</Text>
          <Text size="small" weight="semibold">#{node.id}</Text>
        </div>
      </Grid>
    </Stack>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export default function MCTSTreeExplorer() {
  const theme = useHostTheme();
  const [selectedId, setSelectedId] = useCanvasState<number | null>("selectedId", DEFAULT_SELECTED_ID);
  const selectedNode = selectedId != null ? (nodesMap.get(selectedId) ?? null) : null;

  const allNodes = Array.from(nodesMap.values());
  const totalAudited = allNodes.filter(n => n.audit !== null).length;
  const totalPruned = allNodes.filter(n => n.dead).length;
  const bestLeaf = nodesMap.get(DEFAULT_SELECTED_ID);
  const bestScore = bestLeaf ? Math.round(bestLeaf.mean_value * 100) : 0;

  return (
    <Stack gap={0} style={{ height: "100%", overflow: "hidden", background: theme.bg.editor }}>
      {/* Header */}
      <Stack gap={12} style={{ padding: "20px 24px 16px", flexShrink: 0 }}>
        <H1>MCTS Search Tree</H1>
        <Text tone="secondary" size="small">{trunc(TREE_DATA.idea, 130)}</Text>
        <Grid columns={4} gap={12}>
          <Stat value={String(allNodes.length)} label="Total nodes" />
          <Stat value={String(totalAudited)} label="Audited" />
          <Stat value={String(totalPruned)} label="Pruned" tone="warning" />
          <Stat value={`${bestScore}%`} label="Best-path score" tone="success" />
        </Grid>
      </Stack>

      <Divider />

      <Row gap={0} align="stretch" style={{ flex: 1, overflow: "hidden", minHeight: 0 }}>
        {/* Left: scrollable SVG tree */}
        <div style={{ flex: 1, overflow: "auto", padding: 16 }}>
          <svg width={layout.width} height={layout.height} style={{ display: "block" }}>
            {/* Edges */}
            {layout.edges.map(e => {
              const fromId = parseInt(e.from);
              const toId = parseInt(e.to);
              const onBest = BEST_PATH_IDS.has(fromId) && BEST_PATH_IDS.has(toId);
              const midY = (e.sourceY + e.targetY) / 2;
              const d = `M ${e.sourceX} ${e.sourceY} C ${e.sourceX} ${midY}, ${e.targetX} ${midY}, ${e.targetX} ${e.targetY}`;
              return (
                <path
                  key={`e-${e.from}-${e.to}`}
                  d={d}
                  fill="none"
                  stroke={onBest ? theme.accent.primary : theme.stroke.secondary}
                  strokeWidth={onBest ? 2.5 : 1.5}
                />
              );
            })}
            {/* Nodes */}
            {layout.nodes.map(ln => {
              const id = parseInt(ln.id);
              const node = nodesMap.get(id)!;
              const isSelected = selectedId === id;
              const isBest = BEST_PATH_IDS.has(id);
              const isRoot = node.depth === 0;
              const bg = isSelected
                ? theme.fill.secondary
                : isRoot
                ? theme.fill.tertiary
                : theme.bg.chrome;
              return (
                <foreignObject
                  key={`n-${id}`}
                  x={ln.x}
                  y={ln.y}
                  width={NW}
                  height={NH}
                  style={{ cursor: "pointer" }}
                  onClick={() => setSelectedId(id)}
                >
                  <div
                    style={{
                      width: NW,
                      height: NH,
                      background: bg,
                      border: `1px solid ${isBest ? theme.accent.primary : theme.stroke.secondary}`,
                      borderLeft: `3px solid ${isBest ? theme.accent.primary : theme.stroke.secondary}`,
                      borderRadius: 6,
                      padding: "7px 9px",
                      boxSizing: "border-box",
                      overflow: "hidden",
                      opacity: node.dead ? 0.55 : 1,
                      display: "flex",
                      flexDirection: "column",
                      gap: 5,
                    }}
                  >
                    {/* Idea text — pre-truncated, height-clipped */}
                    <div
                      style={{
                        fontSize: 10,
                        lineHeight: 1.4,
                        maxHeight: isRoot ? 28 : 42,
                        overflow: "hidden",
                        color: node.dead ? theme.text.tertiary : theme.text.primary,
                        fontStyle: isRoot ? "italic" : undefined,
                        textDecoration: node.dead ? "line-through" : undefined,
                      }}
                    >
                      {trunc(node.idea, isRoot ? 85 : 120)}
                    </div>
                    {/* Bottom row: check badges + score */}
                    <div
                      style={{
                        display: "flex",
                        gap: 3,
                        alignItems: "center",
                        marginTop: "auto",
                      }}
                    >
                      {node.audit
                        ? CHECKS.map(k => (
                            <span
                              key={k}
                              title={`${CHECK_LABELS[k]}: ${auditPassed(node.audit!, k) ? "PASS" : "FAIL"}`}
                              style={{
                                fontSize: 9,
                                padding: "1px 3px",
                                borderRadius: 3,
                                background: auditPassed(node.audit!, k)
                                  ? theme.diff.insertedLine
                                  : theme.diff.removedLine,
                                color: auditPassed(node.audit!, k)
                                  ? theme.diff.stripAdded
                                  : theme.diff.stripRemoved,
                                fontWeight: 700,
                                letterSpacing: 0.3,
                              }}
                            >
                              {k[0].toUpperCase()}
                            </span>
                          ))
                        : null}
                      <span
                        style={{
                          marginLeft: "auto",
                          fontSize: 10,
                          color: theme.text.secondary,
                          fontWeight: 600,
                        }}
                      >
                        {Math.round(node.mean_value * 100)}%
                      </span>
                      {node.dead && (
                        <span style={{ fontSize: 9, color: theme.text.tertiary, marginLeft: 3 }}>
                          pruned
                        </span>
                      )}
                    </div>
                  </div>
                </foreignObject>
              );
            })}
          </svg>
        </div>

        {/* Right: detail panel */}
        <div
          style={{
            width: 380,
            flexShrink: 0,
            borderLeft: `1px solid ${theme.stroke.tertiary}`,
            overflow: "auto",
            padding: "16px 20px",
          }}
        >
          {selectedNode == null ? (
            <Text tone="tertiary" size="small">Click a node in the tree to inspect it.</Text>
          ) : (
            <NodeDetail node={selectedNode} theme={theme} />
          )}
        </div>
      </Row>
    </Stack>
  );
}
"""


def _compute_best_path_ids(tree: dict) -> list[int]:
    """Walk a tree dict following the highest mean_value child at each step."""
    path = [tree["id"]]
    cur = tree
    while cur.get("children"):
        live = [c for c in cur["children"] if not c.get("dead", False)]
        candidates = live or cur["children"]
        if not candidates:
            break
        best = max(candidates, key=lambda c: (c["mean_value"], c["visits"]))
        path.append(best["id"])
        cur = best
    return path


def _canvas_default_out(cwd: Path) -> Path:
    """Return the Cursor-managed canvases directory for ``cwd``."""
    path_str = str(cwd)
    if len(path_str) >= 2 and path_str[1] == ":":
        path_str = path_str[0].lower() + path_str[2:]
    slug = path_str.replace("\\", "/").replace("/", "-").strip("-")
    return Path.home() / ".cursor" / "projects" / slug / "canvases"


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MCTS Tree Explorer</title>
<style>
:root {
  --bg:#1e1e1e;--bg-c:#252526;--bg-s:#2d2d2d;--tx:#d4d4d4;--tx2:#9d9d9d;
  --tx3:#6e6e6e;--st:#3e3e3e;--st2:#4a4a4a;--ac:#569cd6;
  --ok:#4ec9b0;--warn:#ce9178;--err:#f44747;
  --ok-bg:rgba(78,201,176,.15);--err-bg:rgba(244,71,71,.15);--warn-bg:rgba(206,145,120,.12);
}
@media(prefers-color-scheme:light){:root{
  --bg:#fff;--bg-c:#f3f3f3;--bg-s:#e8e8e8;--tx:#1e1e1e;--tx2:#616161;
  --tx3:#9e9e9e;--st:#e0e0e0;--st2:#d0d0d0;--ac:#0066b8;
  --ok:#0a7c3e;--warn:#795e26;--err:#a31515;
  --ok-bg:rgba(10,124,62,.1);--err-bg:rgba(163,21,21,.1);--warn-bg:rgba(121,94,38,.1);
}}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--tx);height:100vh;overflow:hidden;display:flex;flex-direction:column}
#hdr{padding:16px 24px 12px;flex-shrink:0;border-bottom:1px solid var(--st)}
h1{font-size:20px;font-weight:600;margin-bottom:4px}
#prob{font-size:12px;color:var(--tx2);margin-bottom:12px}
#stats{display:flex;gap:24px}
.sv{font-size:20px;font-weight:600}.sl{font-size:11px;color:var(--tx2);margin-top:1px}
.sv.ok{color:var(--ok)}.sv.warn{color:var(--warn)}
#main{display:flex;flex:1;overflow:hidden;min-height:0}
#tree{flex:1;overflow:auto;padding:16px}
#tree svg{display:block}
#detail{width:380px;flex-shrink:0;border-left:1px solid var(--st);overflow:auto;padding:16px 20px}
.nb{cursor:pointer;border:1px solid var(--st2);border-left:3px solid var(--st2);border-radius:6px;padding:7px 9px;overflow:hidden;display:flex;flex-direction:column;gap:5px;background:var(--bg-c)}
.nb.sel{background:var(--bg-s)}.nb.bp{border-color:var(--ac)}.nb.dead{opacity:.55}
.ni{font-size:10px;line-height:1.4;overflow:hidden;color:var(--tx)}
.ni.root{font-style:italic}.ni.dead{text-decoration:line-through;color:var(--tx3)}
.nb-row{display:flex;gap:3px;align-items:center;margin-top:auto}
.badge{font-size:9px;padding:1px 3px;border-radius:3px;font-weight:700;letter-spacing:.3px}
.badge.ok{background:var(--ok-bg);color:var(--ok)}.badge.err{background:var(--err-bg);color:var(--err)}
.nscore{margin-left:auto;font-size:10px;color:var(--tx2);font-weight:600}
.npruned{font-size:9px;color:var(--tx3);margin-left:3px}
#detail h3{font-size:13px;font-weight:600;margin:12px 0 6px}
.dstats{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:4px}
.dsv{font-size:20px;font-weight:600}.dsl{font-size:11px;color:var(--tx2)}
.dsv.ok{color:var(--ok)}.dsv.err{color:var(--err)}
.callout{padding:8px 12px;border-radius:4px;border-left:3px solid;margin:8px 0;font-size:12px}
.ct{font-weight:600;margin-bottom:2px;font-size:12px}
.callout.ok{border-color:var(--ok);background:var(--ok-bg)}
.callout.warn{border-color:var(--warn);background:var(--warn-bg)}
.didea{font-size:12px;line-height:1.6}
hr{border:none;border-top:1px solid var(--st);margin:12px 0}
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;font-size:11px;color:var(--tx2);padding:4px 6px;border-bottom:1px solid var(--st)}
td{padding:5px 6px;vertical-align:top;border-bottom:1px solid var(--st)}
tr.ok td:nth-child(2){color:var(--ok);font-weight:600}
tr.err td:nth-child(2){color:var(--err);font-weight:600}
.dmeta{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
.ml{font-size:11px;color:var(--tx2)}.mv{font-size:12px;font-weight:600}
.ph{color:var(--tx3);font-size:12px}
</style>
</head>
<body>
<div id="hdr">
  <h1>MCTS Search Tree</h1>
  <div id="prob"></div>
  <div id="stats"></div>
</div>
<div id="main">
  <div id="tree"><svg id="svg"></svg></div>
  <div id="detail"><p class="ph">Click a node in the tree to inspect it.</p></div>
</div>
<script>
const T=__TREE_JSON__;
const BP=new Set([__BEST_PATH_IDS__]);
const DID=__DEFAULT_SELECTED_ID__;
const NW=240,NH=100,RG=90,NG=24,PAD=32;
const nm=new Map(),edges=[];
(function flat(n){nm.set(n.id,n);for(const c of n.children){edges.push({f:n.id,t:c.id});flat(c)}})(T);
function countLeaves(n){if(!n.children||!n.children.length)return 1;return n.children.reduce((s,c)=>s+countLeaves(c),0)}
const pos=new Map();
function layout(n,left,d){
  const lv=countLeaves(n),w=lv*(NW+NG)-NG;
  pos.set(n.id,{x:left+(w-NW)/2,y:PAD+d*(NH+RG)});
  let cl=left;
  for(const c of n.children){const cv=countLeaves(c);layout(c,cl,d+1);cl+=cv*(NW+NG);}
}
layout(T,PAD,0);
let maxX=0,maxY=0;
for(const p of pos.values()){maxX=Math.max(maxX,p.x+NW);maxY=Math.max(maxY,p.y+NH);}
const W=maxX+PAD,H=maxY+PAD;
const NS='http://www.w3.org/2000/svg';
function se(tag,a={}){const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(a))e.setAttribute(k,v);return e}
function trunc(s,n){s=s.replace(/\s+/g,' ').trim();return s.length<=n?s:s.slice(0,n-1)+'\u2026'}
const CK=['novelty','resource','feasibility','alignment'];
const CL={novelty:'Novelty',resource:'Resource',feasibility:'Feasibility',alignment:'Alignment'};
let sel=DID;
function render(){
  const svg=document.getElementById('svg');
  svg.setAttribute('width',W);svg.setAttribute('height',H);svg.innerHTML='';
  for(const e of edges){
    const fp=pos.get(e.f),tp=pos.get(e.t);
    const sx=fp.x+NW/2,sy=fp.y+NH,tx=tp.x+NW/2,ty=tp.y,my=(sy+ty)/2;
    const ob=BP.has(e.f)&&BP.has(e.t);
    svg.appendChild(se('path',{d:`M${sx} ${sy}C${sx} ${my},${tx} ${my},${tx} ${ty}`,fill:'none',
      stroke:ob?'var(--ac)':'var(--st2)','stroke-width':ob?2.5:1.5}));
  }
  for(const[id,n]of nm){
    const p=pos.get(id);
    const fo=se('foreignObject',{x:p.x,y:p.y,width:NW,height:NH});
    const box=document.createElement('div');
    box.className='nb'+(BP.has(id)?' bp':'')+(id===sel?' sel':'')+(n.dead?' dead':'');
    box.style.cssText=`width:${NW}px;height:${NH}px`;
    const ir=n.depth===0;
    const nd=document.createElement('div');
    nd.className='ni'+(ir?' root':'')+(n.dead?' dead':'');
    nd.style.maxHeight=ir?'28px':'42px';
    nd.textContent=trunc(n.idea,ir?85:120);
    box.appendChild(nd);
    const row=document.createElement('div');row.className='nb-row';
    if(n.audit){for(const k of CK){const b=document.createElement('span');b.className='badge '+(n.audit[k]?'ok':'err');b.title=`${CL[k]}: ${n.audit[k]?'PASS':'FAIL'}`;b.textContent=k[0].toUpperCase();row.appendChild(b)}}
    const sc=document.createElement('span');sc.className='nscore';sc.textContent=Math.round(n.mean_value*100)+'%';row.appendChild(sc);
    if(n.dead){const pr=document.createElement('span');pr.className='npruned';pr.textContent='pruned';row.appendChild(pr)}
    box.appendChild(row);
    box.onclick=()=>{sel=id;render();showDetail(n)};
    fo.appendChild(box);svg.appendChild(fo);
  }
}
function showDetail(n){
  const d=document.getElementById('detail');d.innerHTML='';
  const sc=Math.round(n.mean_value*100);
  const ds=document.createElement('div');ds.className='dstats';
  ds.innerHTML=`<div><div class="dsv">${n.visits}</div><div class="dsl">Visits</div></div>`+
    `<div><div class="dsv ${sc>=75?'ok':sc<50?'err':''}">${sc}%</div><div class="dsl">Mean score</div></div>`;
  d.appendChild(ds);
  if(n.dead){const c=document.createElement('div');c.className='callout warn';c.innerHTML='<div class="ct">Pruned</div>Failed a hard-threshold check and removed from further expansion.';d.appendChild(c)}
  if(BP.has(n.id)&&n.depth>0){const c=document.createElement('div');c.className='callout ok';c.innerHTML='<div class="ct">On the best path</div>Lies on the greedy best path from root to leaf.';d.appendChild(c)}
  const h=document.createElement('h3');h.textContent='Idea';d.appendChild(h);
  const p=document.createElement('p');p.className='didea';p.textContent=n.idea;d.appendChild(p);
  if(n.audit){
    d.appendChild(document.createElement('hr'));
    const ah=document.createElement('h3');ah.textContent='QCM Audit';d.appendChild(ah);
    const tbl=document.createElement('table');
    tbl.innerHTML='<thead><tr><th>Check</th><th>Result</th><th>Reason</th></tr></thead>';
    const tb=document.createElement('tbody');
    for(const k of CK){
      const pass=n.audit[k];const tr=document.createElement('tr');tr.className=pass?'ok':'err';
      const reason=(n.audit.reasons&&n.audit.reasons[k])||'\u2014';
      tr.innerHTML=`<td>${CL[k]}</td><td>${pass?'PASS':'FAIL'}</td><td>${reason}</td>`;
      tb.appendChild(tr);
    }
    tbl.appendChild(tb);d.appendChild(tbl);
  }else{
    const p2=document.createElement('p');p2.style.cssText='color:var(--tx3);font-size:12px;font-style:italic;margin-top:8px';
    p2.textContent='Root node — no QCM audit (this is the problem statement).';d.appendChild(p2);
  }
  d.appendChild(document.createElement('hr'));
  const m=document.createElement('div');m.className='dmeta';
  m.innerHTML=`<div><div class="ml">Depth</div><div class="mv">${n.depth}</div></div>`+
    `<div><div class="ml">Value sum</div><div class="mv">${n.value_sum.toFixed(2)}</div></div>`+
    `<div><div class="ml">Node ID</div><div class="mv">#${n.id}</div></div>`;
  d.appendChild(m);
}
const all=Array.from(nm.values());
document.getElementById('prob').textContent=trunc(T.idea,130);
document.getElementById('stats').innerHTML=
  `<div class="stat"><div class="sv">${all.length}</div><div class="sl">Total nodes</div></div>`+
  `<div class="stat"><div class="sv">${all.filter(n=>n.audit).length}</div><div class="sl">Audited</div></div>`+
  `<div class="stat"><div class="sv warn">${all.filter(n=>n.dead).length}</div><div class="sl">Pruned</div></div>`+
  `<div class="stat"><div class="sv ok">${nm.has(DID)?Math.round(nm.get(DID).mean_value*100):0}%</div><div class="sl">Best-path score</div></div>`;
render();
const bl=nm.get(DID);if(bl)showDetail(bl);
</script>
</body>
</html>"""


def generate_html(tree_data: dict, html_out: Path) -> Path:
    """Write a self-contained HTML tree explorer for ``tree_data``.

    The file embeds all data and logic inline — no network calls, no
    dependencies.  Open it in any browser.  Re-run after ``mcts run``
    to get fresh output.

    Args:
        tree_data: The raw dict from ``tree.json``.
        html_out:  Full path (including ``.html`` filename) to write.

    Returns:
        The resolved output path.
    """
    json_str = json.dumps(tree_data, ensure_ascii=False)
    best_path_ids = _compute_best_path_ids(tree_data)
    default_id = best_path_ids[-1]

    html = (
        _HTML_TEMPLATE
        .replace("__TREE_JSON__", json_str)
        .replace("__BEST_PATH_IDS__", ", ".join(str(i) for i in best_path_ids))
        .replace("__DEFAULT_SELECTED_ID__", str(default_id))
    )

    html_out = Path(html_out)
    html_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.write_text(html, encoding="utf-8")
    return html_out.resolve()


def generate_canvas(tree_data: dict, canvas_out: Path) -> Path:
    """Write a Cursor Canvas `.canvas.tsx` file for ``tree_data``.

    The canvas is self-contained: tree data is base-64 encoded and inlined
    directly in the TypeScript so no network call or file read is needed at
    runtime.  Calling this again after a new ``mcts run`` regenerates the file
    with fresh data; the IDE hot-reloads it automatically.

    Args:
        tree_data: The raw dict from ``tree.json`` (i.e. the root node dict).
        canvas_out: Full path *including filename* of the ``.canvas.tsx`` to write.

    Returns:
        The resolved canvas path.
    """
    json_bytes = json.dumps(tree_data, ensure_ascii=True).encode()
    b64 = base64.b64encode(json_bytes).decode()
    best_path_ids = _compute_best_path_ids(tree_data)
    default_id = best_path_ids[-1]

    tsx = (
        _CANVAS_TEMPLATE
        .replace("__TREE_JSON_B64__", b64)
        .replace("__BEST_PATH_IDS__", ", ".join(str(i) for i in best_path_ids))
        .replace("__DEFAULT_SELECTED_ID__", str(default_id))
    )

    canvas_out = Path(canvas_out)
    canvas_out.parent.mkdir(parents=True, exist_ok=True)
    canvas_out.write_text(tsx, encoding="utf-8")
    return canvas_out.resolve()
