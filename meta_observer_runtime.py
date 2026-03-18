"""
Meta-Observer Runtime — Universal Agent Learning Standard
==========================================================
Wraps ANY agent loop with persistent 12D meta-observation,
experience accumulation, cross-agent learning, and environmental
analysis. Adapted from Athena's 57-cycle meta-observation protocol
fused with Karpathy's autoresearch autonomous loop pattern.

DESIGN PRINCIPLE:
  Karpathy's loop: modify → train → evaluate → keep/discard (stateless)
  Meta-Observer:   modify → train → evaluate → OBSERVE → LEARN → REMEMBER → ADAPT

The observer doesn't just track WHAT happened — it tracks WHY,
extracts patterns across cycles, and evolves its own strategy.

USAGE (Standard Interface — works with any agent):
    from meta_observer_runtime import MetaObserver

    observer = MetaObserver(agent_id="autoresearch-001", project="nanochat")

    for cycle in observer.loop():          # infinite loop with cycle tracking
        idea = observer.suggest_next()     # strategy-informed suggestion
        result = run_experiment(idea)      # your agent's work
        observer.observe(idea, result)     # 12D observation + learning
        observer.decide(result)            # keep/discard + memory update

    # Observer automatically:
    #   - Logs every cycle with 12D scores
    #   - Extracts patterns (what types of changes work)
    #   - Tracks velocity/acceleration of improvement
    #   - Learns from other agents' logs (cross-agent)
    #   - Analyzes environment for deeper improvements
    #   - Evolves its own strategy over time
"""

import json
import time
import hashlib
import os
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Iterator
from collections import defaultdict
import math


# ──────────────────────────────────────────────────────────────
#  12-Dimensional Observation Space (from Athena meta-observer)
# ──────────────────────────────────────────────────────────────

DIMENSIONS = {
    "x1_structure":       "Architecture changes, hierarchy, topology, scaffold coherence",
    "x2_semantics":       "Meaning precision, naming quality, conceptual clarity",
    "x3_coordination":    "Agent alignment, handoff quality, shared motion",
    "x4_recursion":       "Learning from prior cycles, compounding gains, closing loops",
    "x5_contradiction":   "Conflict detection, assumption clashes, incoherence",
    "x6_emergence":       "Novel structures, unexpected improvements, pattern formation",
    "x7_legibility":      "Readability, reviewability, handoff clarity",
    "x8_routing":         "Task flow clarity, decision quality, assignment specificity",
    "x9_grounding":       "Evidence quality, metric-backed claims, reproducibility",
    "x10_compression":    "Signal density, redundancy reduction, elegance",
    "x11_interop":        "Cross-system compatibility, module cooperation",
    "x12_potential":      "Future leverage, extensibility, next-step opportunity",
}

# Cross-dimensional coupling matrix (positive/negative interactions)
COUPLING_MATRIX = {
    ("x1_structure", "x7_legibility"):    +0.8,   # better structure → better readability
    ("x8_routing", "x3_coordination"):    +0.7,   # better routing → better coordination
    ("x4_recursion", "x6_emergence"):     +0.6,   # learning → emergence
    ("x10_compression", "x9_grounding"):  -0.3,   # over-compression harms evidence
    ("x5_contradiction", "x2_semantics"): -0.5,   # unresolved conflicts degrade clarity
    ("x6_emergence", "x12_potential"):    +0.9,   # emergence creates future leverage
    ("x9_grounding", "x4_recursion"):     +0.7,   # good evidence enables learning
    ("x3_coordination", "x11_interop"):   +0.6,   # coordination enables interop
}


# ──────────────────────────────────────────────────────────────
#  4-Element Deep Synthesis Lenses (from Athena spec Phase B)
# ──────────────────────────────────────────────────────────────

ELEMENT_LENSES = {
    "Earth": {
        "emphasizes": ["x1_structure", "x7_legibility", "x8_routing", "x10_compression", "x11_interop"],
        "weights": [1.5, 1.3, 1.2, 1.1, 1.2],
        "focus": "topology, structure, scaffolding, material integrity",
    },
    "Fire": {
        "emphasizes": ["x5_contradiction", "x3_coordination", "x4_recursion", "x9_grounding"],
        "weights": [1.5, 1.3, 1.2, 1.4],
        "focus": "conflict, pressure, brittle structures, transformation",
    },
    "Water": {
        "emphasizes": ["x2_semantics", "x3_coordination", "x6_emergence", "x11_interop", "x12_potential"],
        "weights": [1.3, 1.2, 1.5, 1.1, 1.3],
        "focus": "flow, adaptation, connection, meaning",
    },
    "Air": {
        "emphasizes": ["x2_semantics", "x4_recursion", "x6_emergence", "x10_compression", "x12_potential"],
        "weights": [1.2, 1.4, 1.3, 1.5, 1.4],
        "focus": "abstraction, pattern, meta-organization, compression",
    },
}

# Riemannian metric tensor diagonal (from Athena spec)
# Higher weight = more important dimension
METRIC_TENSOR_DIAG = [
    1.3,  # x1: structure
    1.2,  # x2: semantics
    1.4,  # x3: coordination
    1.2,  # x4: recursion
    1.4,  # x5: contradiction
    1.1,  # x6: emergence
    1.2,  # x7: legibility
    1.3,  # x8: routing
    1.4,  # x9: grounding
    1.1,  # x10: compression
    1.2,  # x11: interop
    1.0,  # x12: potential
]


def apply_lens(scores: dict, lens: str) -> dict:
    """Apply an element lens weighting to 12D scores."""
    lens_spec = ELEMENT_LENSES.get(lens, {})
    emphasized = lens_spec.get("emphasizes", [])
    weights = lens_spec.get("weights", [])
    weighted_scores = dict(scores)
    for dim, w in zip(emphasized, weights):
        if dim in weighted_scores:
            weighted_scores[dim] = min(1.0, weighted_scores[dim] * w)
    return weighted_scores


def riemannian_distance(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute Riemannian distance using the metric tensor G."""
    diff = [a - b for a, b in zip(vec_a, vec_b)]
    return math.sqrt(sum(g * d**2 for g, d in zip(METRIC_TENSOR_DIAG, diff)))


def riemannian_magnitude(vec: list[float]) -> float:
    """Compute Riemannian magnitude: sqrt(v^T G v)."""
    return math.sqrt(sum(g * x**2 for g, x in zip(METRIC_TENSOR_DIAG, vec)))


def propagate_coupling(scores: dict) -> dict:
    """
    Apply cross-dimensional coupling matrix.
    When one dimension changes, coupled dimensions are affected.
    """
    updated = dict(scores)
    for (source, target), strength in COUPLING_MATRIX.items():
        if source in updated and target in updated:
            # Coupling: if source deviates from 0.5 (neutral), push target
            deviation = updated[source] - 0.5
            influence = deviation * strength * 0.3  # damped influence
            updated[target] = max(0.0, min(1.0, updated[target] + influence))
    return updated


@dataclass
class Observation:
    """A single 12D observation of one cycle."""
    cycle_id: int
    timestamp: str
    agent_id: str
    project: str

    # What happened
    action_type: str = ""          # e.g., "hyperparameter", "architecture", "optimizer"
    action_description: str = ""   # human-readable description
    action_diff: str = ""          # actual code/config diff

    # Result
    metric_name: str = ""          # e.g., "val_bpb"
    metric_value: float = 0.0
    metric_delta: float = 0.0      # change from previous
    outcome: str = ""              # "keep", "discard", "crash", "neutral"

    # 12D scores (0.0 to 1.0)
    x1_structure: float = 0.5
    x2_semantics: float = 0.5
    x3_coordination: float = 0.5
    x4_recursion: float = 0.5
    x5_contradiction: float = 0.5
    x6_emergence: float = 0.5
    x7_legibility: float = 0.5
    x8_routing: float = 0.5
    x9_grounding: float = 0.5
    x10_compression: float = 0.5
    x11_interop: float = 0.5
    x12_potential: float = 0.5

    # Derived calculus (computed automatically)
    velocity: float = 0.0          # V(n) = X(n) - X(n-1)
    acceleration: float = 0.0     # A(n) = V(n) - V(n-1)
    jerk: float = 0.0             # J(n) = A(n) - A(n-1)

    # Meta-learning tags
    tags: str = "[]"               # JSON-encoded list of tags
    pattern_match: str = ""        # matched pattern from experience memory
    strategy_used: str = ""        # which strategy was active
    environment_state: str = "{}"  # JSON-encoded env snapshot

    # Witness hash (chain integrity)
    prev_hash: str = ""
    obs_hash: str = ""

    def score_vector(self) -> list[float]:
        """Return 12D score as a vector."""
        return [
            self.x1_structure, self.x2_semantics, self.x3_coordination,
            self.x4_recursion, self.x5_contradiction, self.x6_emergence,
            self.x7_legibility, self.x8_routing, self.x9_grounding,
            self.x10_compression, self.x11_interop, self.x12_potential,
        ]

    def magnitude(self) -> float:
        """Riemannian magnitude of the 12D observation vector (weighted by metric tensor G)."""
        return riemannian_magnitude(self.score_vector())


@dataclass
class Pattern:
    """An extracted pattern from accumulated experience."""
    pattern_id: str
    action_type: str
    description: str
    success_rate: float          # 0.0 to 1.0
    avg_improvement: float       # average metric delta when successful
    sample_count: int            # how many observations back this up
    confidence: float            # statistical confidence
    context_conditions: str      # JSON: when does this pattern apply?
    first_seen: str              # timestamp
    last_seen: str               # timestamp
    times_used: int = 0          # how many times strategy engine used this
    times_successful: int = 0    # how many times it actually worked


@dataclass
class EnvironmentSnapshot:
    """Snapshot of the agent's operating environment."""
    timestamp: str
    gpu_util: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    cpu_util: float = 0.0
    disk_free_gb: float = 0.0
    active_agents: int = 1
    current_best_metric: float = float('inf')
    total_experiments: int = 0
    experiments_since_improvement: int = 0
    exploration_ratio: float = 1.0  # 1.0=pure explore, 0.0=pure exploit
    custom: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────
#  Experience Memory — Persistent Learning Store
# ──────────────────────────────────────────────────────────────

class ExperienceMemory:
    """
    SQLite-backed persistent memory that accumulates observations,
    extracts patterns, and enables cross-agent learning.

    This is what makes the meta-observer LEARN rather than just OBSERVE.
    """

    def __init__(self, db_path: str = "meta_observer.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        c = self.conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS observations (
                cycle_id INTEGER,
                timestamp TEXT,
                agent_id TEXT,
                project TEXT,
                action_type TEXT,
                action_description TEXT,
                action_diff TEXT,
                metric_name TEXT,
                metric_value REAL,
                metric_delta REAL,
                outcome TEXT,
                x1 REAL, x2 REAL, x3 REAL, x4 REAL, x5 REAL, x6 REAL,
                x7 REAL, x8 REAL, x9 REAL, x10 REAL, x11 REAL, x12 REAL,
                velocity REAL, acceleration REAL, jerk REAL,
                tags TEXT,
                pattern_match TEXT,
                strategy_used TEXT,
                environment_state TEXT,
                prev_hash TEXT,
                obs_hash TEXT,
                PRIMARY KEY (agent_id, cycle_id)
            );

            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id TEXT PRIMARY KEY,
                action_type TEXT,
                description TEXT,
                success_rate REAL,
                avg_improvement REAL,
                sample_count INTEGER,
                confidence REAL,
                context_conditions TEXT,
                first_seen TEXT,
                last_seen TEXT,
                times_used INTEGER DEFAULT 0,
                times_successful INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS agent_registry (
                agent_id TEXT PRIMARY KEY,
                project TEXT,
                created TEXT,
                last_active TEXT,
                total_cycles INTEGER DEFAULT 0,
                best_metric REAL,
                best_metric_cycle INTEGER,
                strategy_profile TEXT
            );

            CREATE TABLE IF NOT EXISTS cross_agent_insights (
                insight_id TEXT PRIMARY KEY,
                source_agent TEXT,
                target_agent TEXT,
                insight_type TEXT,
                description TEXT,
                evidence TEXT,
                timestamp TEXT,
                applied INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS environment_log (
                timestamp TEXT,
                agent_id TEXT,
                cycle_id INTEGER,
                gpu_util REAL,
                gpu_memory_used_mb REAL,
                cpu_util REAL,
                current_best REAL,
                total_experiments INTEGER,
                experiments_since_improvement INTEGER,
                exploration_ratio REAL,
                custom TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_obs_agent ON observations(agent_id);
            CREATE INDEX IF NOT EXISTS idx_obs_outcome ON observations(outcome);
            CREATE INDEX IF NOT EXISTS idx_obs_type ON observations(action_type);
            CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(action_type);
            CREATE INDEX IF NOT EXISTS idx_patterns_success ON patterns(success_rate DESC);

            CREATE TABLE IF NOT EXISTS contradictions (
                contradiction_id TEXT PRIMARY KEY,
                agent_id TEXT,
                cycle_detected INTEGER,
                description TEXT,
                dimension_a TEXT,
                dimension_b TEXT,
                severity REAL,
                status TEXT DEFAULT 'open',
                resolution TEXT DEFAULT '',
                resolved_cycle INTEGER DEFAULT 0,
                timestamp TEXT
            );

            CREATE TABLE IF NOT EXISTS emergence_events (
                event_id TEXT PRIMARY KEY,
                agent_id TEXT,
                cycle_id INTEGER,
                description TEXT,
                dimensions_affected TEXT,
                coherence_gain REAL,
                novelty_score REAL,
                reuse_potential REAL,
                reproducible INTEGER DEFAULT 0,
                promoted_to_pattern INTEGER DEFAULT 0,
                timestamp TEXT
            );

            CREATE TABLE IF NOT EXISTS strategy_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                cycle_id INTEGER,
                strategy TEXT,
                outcome TEXT,
                metric_delta REAL,
                timestamp TEXT
            );
        """)
        self.conn.commit()

    def store_observation(self, obs: Observation):
        """Store a single observation."""
        c = self.conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO observations VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?
            )
        """, (
            obs.cycle_id, obs.timestamp, obs.agent_id, obs.project,
            obs.action_type, obs.action_description, obs.action_diff,
            obs.metric_name, obs.metric_value, obs.metric_delta, obs.outcome,
            obs.x1_structure, obs.x2_semantics, obs.x3_coordination,
            obs.x4_recursion, obs.x5_contradiction, obs.x6_emergence,
            obs.x7_legibility, obs.x8_routing, obs.x9_grounding,
            obs.x10_compression, obs.x11_interop, obs.x12_potential,
            obs.velocity, obs.acceleration, obs.jerk,
            obs.tags, obs.pattern_match, obs.strategy_used,
            obs.environment_state, obs.prev_hash, obs.obs_hash,
        ))
        self.conn.commit()

    def get_recent_observations(self, agent_id: str, limit: int = 50) -> list[dict]:
        """Get recent observations for an agent."""
        c = self.conn.cursor()
        rows = c.execute(
            "SELECT * FROM observations WHERE agent_id=? ORDER BY cycle_id DESC LIMIT ?",
            (agent_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_observations_by_type(self, action_type: str) -> list[dict]:
        """Get all observations of a given type across ALL agents."""
        c = self.conn.cursor()
        rows = c.execute(
            "SELECT * FROM observations WHERE action_type=? ORDER BY timestamp",
            (action_type,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_success_rate_by_type(self) -> dict[str, dict]:
        """Compute success rates by action type across all agents."""
        c = self.conn.cursor()
        rows = c.execute("""
            SELECT action_type,
                   COUNT(*) as total,
                   SUM(CASE WHEN outcome='keep' THEN 1 ELSE 0 END) as successes,
                   AVG(CASE WHEN outcome='keep' THEN metric_delta ELSE NULL END) as avg_improvement,
                   AVG(CASE WHEN outcome='discard' THEN metric_delta ELSE NULL END) as avg_regression
            FROM observations
            WHERE action_type != ''
            GROUP BY action_type
            ORDER BY total DESC
        """).fetchall()
        return {
            r["action_type"]: {
                "total": r["total"],
                "successes": r["successes"],
                "success_rate": r["successes"] / r["total"] if r["total"] > 0 else 0,
                "avg_improvement": r["avg_improvement"] or 0,
                "avg_regression": r["avg_regression"] or 0,
            }
            for r in rows
        }

    def extract_patterns(self, min_samples: int = 3) -> list[Pattern]:
        """
        Extract patterns from accumulated observations.
        This is the LEARNING step — find what works, when, and why.
        """
        stats = self.get_success_rate_by_type()
        patterns = []

        for action_type, s in stats.items():
            if s["total"] < min_samples:
                continue

            # Compute confidence using Wilson score interval
            n = s["total"]
            p = s["success_rate"]
            z = 1.96  # 95% confidence
            denominator = 1 + z**2 / n
            centre = (p + z**2 / (2*n)) / denominator
            margin = z * math.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator
            confidence = max(0, centre - margin)

            # Find context conditions (when does this type of action work best?)
            context = self._analyze_context(action_type)

            # Get time range
            c = self.conn.cursor()
            time_range = c.execute(
                "SELECT MIN(timestamp) as first, MAX(timestamp) as last FROM observations WHERE action_type=?",
                (action_type,)
            ).fetchone()

            pattern = Pattern(
                pattern_id=hashlib.md5(f"{action_type}:{n}".encode()).hexdigest()[:12],
                action_type=action_type,
                description=f"{action_type}: {s['successes']}/{s['total']} successful "
                           f"(avg improvement: {s['avg_improvement']:.6f})",
                success_rate=p,
                avg_improvement=s["avg_improvement"],
                sample_count=n,
                confidence=confidence,
                context_conditions=json.dumps(context),
                first_seen=time_range["first"] or "",
                last_seen=time_range["last"] or "",
            )
            patterns.append(pattern)

            # Store/update pattern in DB
            c.execute("""
                INSERT OR REPLACE INTO patterns VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                pattern.pattern_id, pattern.action_type, pattern.description,
                pattern.success_rate, pattern.avg_improvement, pattern.sample_count,
                pattern.confidence, pattern.context_conditions,
                pattern.first_seen, pattern.last_seen,
                pattern.times_used, pattern.times_successful,
            ))
        self.conn.commit()
        return patterns

    def _analyze_context(self, action_type: str) -> dict:
        """Analyze when a particular action type tends to succeed vs fail."""
        c = self.conn.cursor()

        # Compare successful vs failed experiments of this type
        success_metrics = c.execute("""
            SELECT AVG(metric_value) as avg_metric,
                   AVG(x4) as avg_recursion,
                   AVG(x6) as avg_emergence,
                   AVG(x12) as avg_potential
            FROM observations
            WHERE action_type=? AND outcome='keep'
        """, (action_type,)).fetchone()

        fail_metrics = c.execute("""
            SELECT AVG(metric_value) as avg_metric,
                   AVG(x4) as avg_recursion
            FROM observations
            WHERE action_type=? AND outcome='discard'
        """, (action_type,)).fetchone()

        context = {
            "success_avg_metric": success_metrics["avg_metric"] if success_metrics else None,
            "fail_avg_metric": fail_metrics["avg_metric"] if fail_metrics else None,
        }

        # Check if this action type works better early or late
        c2 = c.execute("""
            SELECT cycle_id, outcome FROM observations
            WHERE action_type=? ORDER BY cycle_id
        """, (action_type,)).fetchall()

        if len(c2) >= 6:
            mid = len(c2) // 2
            early = [r for r in c2[:mid]]
            late = [r for r in c2[mid:]]
            early_rate = sum(1 for r in early if r["outcome"] == "keep") / len(early) if early else 0
            late_rate = sum(1 for r in late if r["outcome"] == "keep") / len(late) if late else 0
            context["early_success_rate"] = round(early_rate, 3)
            context["late_success_rate"] = round(late_rate, 3)
            context["trend"] = "improving" if late_rate > early_rate + 0.1 else (
                "declining" if early_rate > late_rate + 0.1 else "stable"
            )

        return context

    def record_contradiction(self, agent_id: str, cycle: int,
                              description: str, dim_a: str, dim_b: str,
                              severity: float):
        """Record a detected contradiction between dimensions or expectations."""
        cid = hashlib.md5(f"{agent_id}:{cycle}:{dim_a}:{dim_b}".encode()).hexdigest()[:12]
        c = self.conn.cursor()
        c.execute("""
            INSERT OR IGNORE INTO contradictions VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (cid, agent_id, cycle, description, dim_a, dim_b, severity,
              "open", "", 0, datetime.now(timezone.utc).isoformat()))
        self.conn.commit()
        return cid

    def resolve_contradiction(self, contradiction_id: str, resolution: str, cycle: int):
        """Mark a contradiction as resolved."""
        c = self.conn.cursor()
        c.execute("""
            UPDATE contradictions SET status='resolved', resolution=?, resolved_cycle=?
            WHERE contradiction_id=?
        """, (resolution, cycle, contradiction_id))
        self.conn.commit()

    def get_open_contradictions(self, agent_id: str) -> list[dict]:
        """Get all unresolved contradictions for an agent."""
        c = self.conn.cursor()
        rows = c.execute(
            "SELECT * FROM contradictions WHERE agent_id=? AND status='open' ORDER BY severity DESC",
            (agent_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def record_emergence(self, agent_id: str, cycle: int, description: str,
                         dimensions: list[str], coherence_gain: float,
                         novelty: float, reuse_potential: float):
        """Record an emergence event — unexpected coherent improvement."""
        eid = hashlib.md5(f"{agent_id}:{cycle}:{description[:30]}".encode()).hexdigest()[:12]
        c = self.conn.cursor()
        c.execute("""
            INSERT OR IGNORE INTO emergence_events VALUES (?,?,?,?,?,?,?,?,0,0,?)
        """, (eid, agent_id, cycle, description, json.dumps(dimensions),
              coherence_gain, novelty, reuse_potential,
              datetime.now(timezone.utc).isoformat()))
        self.conn.commit()
        return eid

    def get_emergence_events(self, agent_id: str, limit: int = 20) -> list[dict]:
        """Get recent emergence events."""
        c = self.conn.cursor()
        rows = c.execute(
            "SELECT * FROM emergence_events WHERE agent_id=? ORDER BY coherence_gain DESC LIMIT ?",
            (agent_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]

    def promote_emergence_to_pattern(self, event_id: str):
        """Promote a validated emergence event to a pattern."""
        c = self.conn.cursor()
        c.execute(
            "UPDATE emergence_events SET promoted_to_pattern=1 WHERE event_id=?",
            (event_id,)
        )
        self.conn.commit()

    def record_strategy_outcome(self, agent_id: str, cycle: int,
                                 strategy: str, outcome: str, metric_delta: float):
        """Record strategy effectiveness for meta-learning."""
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO strategy_outcomes (agent_id, cycle_id, strategy, outcome, metric_delta, timestamp)
            VALUES (?,?,?,?,?,?)
        """, (agent_id, cycle, strategy, outcome, metric_delta,
              datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def get_strategy_effectiveness(self, agent_id: str = None) -> dict:
        """Compute strategy effectiveness across all or one agent."""
        c = self.conn.cursor()
        if agent_id:
            rows = c.execute("""
                SELECT strategy, COUNT(*) as total,
                       SUM(CASE WHEN outcome='keep' THEN 1 ELSE 0 END) as successes,
                       AVG(CASE WHEN outcome='keep' THEN metric_delta ELSE NULL END) as avg_imp
                FROM strategy_outcomes WHERE agent_id=?
                GROUP BY strategy ORDER BY total DESC
            """, (agent_id,)).fetchall()
        else:
            rows = c.execute("""
                SELECT strategy, COUNT(*) as total,
                       SUM(CASE WHEN outcome='keep' THEN 1 ELSE 0 END) as successes,
                       AVG(CASE WHEN outcome='keep' THEN metric_delta ELSE NULL END) as avg_imp
                FROM strategy_outcomes GROUP BY strategy ORDER BY total DESC
            """).fetchall()
        return {
            r["strategy"]: {
                "total": r["total"],
                "success_rate": r["successes"] / max(r["total"], 1),
                "avg_improvement": r["avg_imp"] or 0,
            }
            for r in rows
        }

    def get_cross_agent_insights(self, target_agent: str) -> list[dict]:
        """Get insights from other agents that apply to this agent."""
        c = self.conn.cursor()
        rows = c.execute("""
            SELECT * FROM cross_agent_insights
            WHERE target_agent=? AND applied=0
            ORDER BY timestamp DESC
        """, (target_agent,)).fetchall()
        return [dict(r) for r in rows]

    def share_insight(self, source: str, target: str, insight_type: str,
                      description: str, evidence: str):
        """Share a learning from one agent to another."""
        insight_id = hashlib.md5(
            f"{source}:{target}:{insight_type}:{time.time()}".encode()
        ).hexdigest()[:16]
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO cross_agent_insights VALUES (?,?,?,?,?,?,?,0)
        """, (insight_id, source, target, insight_type, description, evidence,
              datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def register_agent(self, agent_id: str, project: str):
        """Register an agent in the agent registry."""
        c = self.conn.cursor()
        c.execute("""
            INSERT OR IGNORE INTO agent_registry (agent_id, project, created, last_active, total_cycles)
            VALUES (?, ?, ?, ?, 0)
        """, (agent_id, project,
              datetime.now(timezone.utc).isoformat(),
              datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def update_agent_stats(self, agent_id: str, cycle: int, best_metric: float):
        """Update agent registry with latest stats."""
        c = self.conn.cursor()
        c.execute("""
            UPDATE agent_registry SET
                last_active=?, total_cycles=?, best_metric=?, best_metric_cycle=?
            WHERE agent_id=?
        """, (datetime.now(timezone.utc).isoformat(), cycle, best_metric, cycle, agent_id))
        self.conn.commit()

    def get_other_agents(self, exclude: str) -> list[dict]:
        """Get all registered agents except the given one."""
        c = self.conn.cursor()
        rows = c.execute(
            "SELECT * FROM agent_registry WHERE agent_id != ?", (exclude,)
        ).fetchall()
        return [dict(r) for r in rows]

    def log_environment(self, env: EnvironmentSnapshot, agent_id: str, cycle_id: int):
        """Log an environment snapshot."""
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO environment_log VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (env.timestamp, agent_id, cycle_id,
              env.gpu_util, env.gpu_memory_used_mb, env.cpu_util,
              env.current_best_metric, env.total_experiments,
              env.experiments_since_improvement, env.exploration_ratio,
              json.dumps(env.custom)))
        self.conn.commit()

    def close(self):
        self.conn.close()


# ──────────────────────────────────────────────────────────────
#  Strategy Engine — Experience-Informed Decision Making
# ──────────────────────────────────────────────────────────────

class StrategyEngine:
    """
    Uses accumulated experience to recommend next actions.
    Implements explore/exploit balance with phi-damped decay.

    Strategy evolution:
      - Early cycles: Pure exploration (try everything)
      - Mid cycles: Pattern-guided exploration (favor what worked)
      - Late cycles: Exploitation + combination (compound known gains)
      - Stagnation: Forced mutation (break out of local optima)
    """

    PHI = (1 + math.sqrt(5)) / 2  # Golden ratio for damping

    def __init__(self, memory: ExperienceMemory, agent_id: str):
        self.memory = memory
        self.agent_id = agent_id
        self._stagnation_counter = 0
        self._last_improvement_cycle = 0
        self._strategy_history: list[dict] = []  # track which strategies worked

    def record_strategy_outcome(self, strategy: str, outcome: str, metric_delta: float):
        """Track which strategies produce improvements — meta-learning about learning."""
        self._strategy_history.append({
            "strategy": strategy,
            "outcome": outcome,
            "metric_delta": metric_delta,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def get_strategy_effectiveness(self) -> dict:
        """Compute success rate per strategy type — learning how to learn."""
        stats = defaultdict(lambda: {"total": 0, "successes": 0, "total_improvement": 0.0})
        for entry in self._strategy_history:
            s = entry["strategy"]
            stats[s]["total"] += 1
            if entry["outcome"] == "keep":
                stats[s]["successes"] += 1
                stats[s]["total_improvement"] += entry["metric_delta"]
        return {
            k: {
                "total": v["total"],
                "success_rate": v["successes"] / max(v["total"], 1),
                "avg_improvement": v["total_improvement"] / max(v["successes"], 1),
            }
            for k, v in stats.items()
        }

    def suggest_action_type(self, cycle: int, current_metric: float) -> dict:
        """
        Suggest what TYPE of action to try next.

        Returns dict with:
          - action_type: str (what kind of change)
          - reasoning: str (why)
          - confidence: float (0-1)
          - strategy: str (explore/exploit/mutate/combine)
        """
        patterns = self.memory.extract_patterns(min_samples=2)
        recent = self.memory.get_recent_observations(self.agent_id, limit=20)
        cross_insights = self.memory.get_cross_agent_insights(self.agent_id)

        # Determine current strategy phase (with metacognition)
        strategy = self._determine_strategy(cycle, recent)

        # Meta-learning: check if this strategy has been working
        strategy_stats = self.get_strategy_effectiveness()
        if strategy in strategy_stats and strategy_stats[strategy]["total"] > 5:
            s = strategy_stats[strategy]
            if s["success_rate"] < 0.1:
                # This strategy consistently fails — switch!
                alternatives = ["explore", "exploit", "combine", "mutate"]
                alternatives.remove(strategy)
                best_alt = max(
                    alternatives,
                    key=lambda a: strategy_stats.get(a, {}).get("success_rate", 0.5)
                )
                strategy = best_alt

        if strategy == "explore":
            return self._suggest_explore(patterns, recent, cross_insights)
        elif strategy == "exploit":
            return self._suggest_exploit(patterns, recent)
        elif strategy == "combine":
            return self._suggest_combine(patterns, recent)
        elif strategy == "mutate":
            return self._suggest_mutate(patterns, recent, cross_insights)
        else:
            return self._suggest_explore(patterns, recent, cross_insights)

    def _determine_strategy(self, cycle: int, recent: list[dict]) -> str:
        """
        Determine which strategy phase we're in.
        Uses phi-damped exploration decay with stagnation detection.
        """
        if cycle < 5:
            return "explore"

        # Count cycles since last improvement
        improvements_in_recent = sum(
            1 for r in recent[:10] if r.get("outcome") == "keep"
        )

        if improvements_in_recent == 0 and len(recent) >= 10:
            self._stagnation_counter += 1
            if self._stagnation_counter >= 3:
                self._stagnation_counter = 0
                return "mutate"  # Break out of local optimum

        if improvements_in_recent > 0:
            self._stagnation_counter = 0

        # Phi-damped exploration ratio
        # explore_ratio = 1/phi^(cycle/10) — decays from 1.0 toward ~0.1
        explore_ratio = 1.0 / (self.PHI ** (cycle / 10))

        if explore_ratio > 0.6:
            return "explore"
        elif explore_ratio > 0.3:
            return "exploit"
        else:
            # Late game: try combining known gains
            return "combine"

    def _suggest_explore(self, patterns: list[Pattern], recent: list[dict],
                         cross_insights: list[dict]) -> dict:
        """Explore: try action types not yet well-sampled."""
        # Find under-explored action types
        explored_types = {p.action_type: p.sample_count for p in patterns}

        # Check cross-agent insights for suggestions
        for insight in cross_insights:
            if insight.get("insight_type") == "successful_strategy":
                return {
                    "action_type": insight.get("description", "unknown"),
                    "reasoning": f"Cross-agent insight from {insight['source_agent']}: {insight['description']}",
                    "confidence": 0.6,
                    "strategy": "explore (cross-agent)",
                }

        # Suggest least-explored type, or a new type
        candidate_types = [
            "hyperparameter_lr", "hyperparameter_batch",
            "architecture_depth", "architecture_width",
            "architecture_attention", "optimizer_config",
            "scheduler_warmup", "scheduler_decay",
            "regularization", "activation_function",
            "normalization", "initialization",
            "embedding_dimension", "window_pattern",
        ]

        unexplored = [t for t in candidate_types if t not in explored_types]
        if unexplored:
            chosen = unexplored[0]
            return {
                "action_type": chosen,
                "reasoning": f"Unexplored action type — gathering baseline data",
                "confidence": 0.3,
                "strategy": "explore",
            }

        # All explored — pick lowest sample count
        least = min(patterns, key=lambda p: p.sample_count)
        return {
            "action_type": least.action_type,
            "reasoning": f"Under-sampled ({least.sample_count} experiments). Need more data.",
            "confidence": 0.4,
            "strategy": "explore",
        }

    def _suggest_exploit(self, patterns: list[Pattern], recent: list[dict]) -> dict:
        """Exploit: favor action types with highest success rate × confidence."""
        if not patterns:
            return self._suggest_explore(patterns, recent, [])

        # Score each pattern: success_rate × confidence × avg_improvement
        scored = []
        for p in patterns:
            if p.avg_improvement <= 0:
                continue
            score = p.success_rate * p.confidence * abs(p.avg_improvement) * 1000
            scored.append((score, p))

        if not scored:
            return self._suggest_explore(patterns, recent, [])

        scored.sort(reverse=True)
        best = scored[0][1]
        return {
            "action_type": best.action_type,
            "reasoning": f"Best performing type: {best.success_rate:.0%} success, "
                        f"avg improvement {best.avg_improvement:.6f} ({best.sample_count} samples)",
            "confidence": best.confidence,
            "strategy": "exploit",
        }

    def _suggest_combine(self, patterns: list[Pattern], recent: list[dict]) -> dict:
        """Combine: try mixing two successful action types."""
        successful = [p for p in patterns if p.success_rate > 0.3 and p.avg_improvement > 0]
        if len(successful) >= 2:
            # Sort by avg improvement, pick top 2
            successful.sort(key=lambda p: p.avg_improvement, reverse=True)
            a, b = successful[0], successful[1]
            return {
                "action_type": f"combine:{a.action_type}+{b.action_type}",
                "reasoning": f"Combine two proven strategies: {a.action_type} "
                            f"({a.avg_improvement:.6f}) + {b.action_type} ({b.avg_improvement:.6f})",
                "confidence": min(a.confidence, b.confidence) * 0.8,
                "strategy": "combine",
            }
        return self._suggest_exploit(patterns, recent)

    def _suggest_mutate(self, patterns: list[Pattern], recent: list[dict],
                        cross_insights: list[dict]) -> dict:
        """Mutate: stagnation detected — try something radical."""
        # Find what we HAVEN'T tried recently
        recent_types = set(r.get("action_type", "") for r in recent[:20])

        radical_ideas = [
            ("architecture_radical", "Major architecture change — break symmetry"),
            ("hyperparameter_extreme", "Extreme hyperparameter value — 10x or 0.1x baseline"),
            ("removal", "Remove a component entirely — test if it's needed"),
            ("reversal", "Reverse a previous improvement — test if context changed"),
            ("import_from_other", "Import strategy from another agent's success"),
        ]

        for idea_type, desc in radical_ideas:
            if idea_type not in recent_types:
                return {
                    "action_type": idea_type,
                    "reasoning": f"STAGNATION BREAK: {desc}",
                    "confidence": 0.2,
                    "strategy": "mutate",
                }

        return {
            "action_type": "random_exploration",
            "reasoning": "Full mutation — all standard approaches exhausted",
            "confidence": 0.1,
            "strategy": "mutate",
        }

    def analyze_environment(self, env: EnvironmentSnapshot) -> dict:
        """
        Analyze the operating environment for deeper improvement opportunities.
        Returns actionable insights about resource utilization, pacing, etc.
        """
        insights = []

        # GPU utilization analysis
        if env.gpu_memory_used_mb > 0 and env.gpu_memory_total_mb > 0:
            gpu_util = env.gpu_memory_used_mb / env.gpu_memory_total_mb
            if gpu_util < 0.5:
                insights.append({
                    "type": "resource_underutilization",
                    "message": f"GPU memory only {gpu_util:.0%} utilized. "
                              f"Consider larger batch size or model.",
                    "priority": "high",
                })
            elif gpu_util > 0.95:
                insights.append({
                    "type": "resource_saturation",
                    "message": "GPU memory near limit. Risk of OOM on larger experiments.",
                    "priority": "medium",
                })

        # Stagnation analysis
        if env.experiments_since_improvement > 15:
            insights.append({
                "type": "deep_stagnation",
                "message": f"{env.experiments_since_improvement} experiments without improvement. "
                          f"Consider fundamental approach change.",
                "priority": "critical",
            })
        elif env.experiments_since_improvement > 8:
            insights.append({
                "type": "mild_stagnation",
                "message": f"{env.experiments_since_improvement} experiments since last improvement. "
                          f"Consider mutation strategy.",
                "priority": "high",
            })

        # Pace analysis
        if env.total_experiments > 0:
            improvement_rate = 1.0 - (env.experiments_since_improvement / max(env.total_experiments, 1))
            insights.append({
                "type": "improvement_rate",
                "message": f"Overall improvement rate: {improvement_rate:.1%}",
                "priority": "info",
            })

        return {
            "insights": insights,
            "recommended_exploration_ratio": self._compute_exploration_ratio(env),
            "environment_health": "healthy" if not any(
                i["priority"] == "critical" for i in insights
            ) else "needs_attention",
        }

    def _compute_exploration_ratio(self, env: EnvironmentSnapshot) -> float:
        """Compute optimal explore/exploit ratio based on environment."""
        if env.total_experiments < 10:
            return 0.9  # Heavy exploration early
        if env.experiments_since_improvement > 10:
            return 0.8  # More exploration when stuck
        # Phi-damped decay
        return max(0.2, 1.0 / (self.PHI ** (env.total_experiments / 20)))


# ──────────────────────────────────────────────────────────────
#  Meta-Observer — The Universal Agent Wrapper
# ──────────────────────────────────────────────────────────────

class MetaObserver:
    """
    Universal meta-observation layer for any autonomous agent.

    Wraps any agent loop with:
      1. Persistent experience memory (SQLite)
      2. 12D observation scoring
      3. Differential calculus (velocity/acceleration/jerk)
      4. Pattern extraction and strategy evolution
      5. Cross-agent learning
      6. Environmental analysis
      7. Witness hash chain (integrity)

    USAGE:
        observer = MetaObserver("agent-001", "my-project")
        for cycle in observer.loop():
            suggestion = observer.suggest_next()
            result = do_work(suggestion)
            observer.observe({"type": "hyperparameter"}, result)
            observer.decide(result)
    """

    def __init__(self, agent_id: str, project: str,
                 db_path: str = "meta_observer.db",
                 log_dir: str = "meta_observer_logs"):
        self.agent_id = agent_id
        self.project = project
        self.cycle = 0
        self.best_metric = float('inf')
        self.best_metric_cycle = 0
        self.experiments_since_improvement = 0

        # Initialize subsystems
        self.memory = ExperienceMemory(db_path)
        self.strategy = StrategyEngine(self.memory, agent_id)

        # Register agent
        self.memory.register_agent(agent_id, project)

        # Log directory
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Observation history (in-memory for calculus)
        self._history: list[Observation] = []
        self._prev_hash = "genesis"

        # Load history from DB if resuming
        recent = self.memory.get_recent_observations(agent_id, limit=100)
        if recent:
            self.cycle = max(r["cycle_id"] for r in recent) + 1
            self._prev_hash = recent[0].get("obs_hash", "genesis")
            # Find best metric
            for r in recent:
                if r.get("outcome") == "keep" and r.get("metric_value", float('inf')) < self.best_metric:
                    self.best_metric = r["metric_value"]
                    self.best_metric_cycle = r["cycle_id"]

        print(f"[MetaObserver] Agent {agent_id} initialized. "
              f"Resuming from cycle {self.cycle}. "
              f"Best metric: {self.best_metric}")

    def loop(self) -> Iterator[int]:
        """
        Infinite generator yielding cycle numbers.
        Use in a for loop — the observer handles everything else.
        """
        while True:
            self.cycle += 1
            yield self.cycle

    def suggest_next(self) -> dict:
        """
        Get a strategy-informed suggestion for the next action.
        Uses accumulated experience, cross-agent insights, and
        environmental analysis.
        """
        env = self._snapshot_environment()
        self.memory.log_environment(env, self.agent_id, self.cycle)

        # Get strategy suggestion
        suggestion = self.strategy.suggest_action_type(
            self.cycle, self.best_metric
        )

        # Get environmental insights
        env_analysis = self.strategy.analyze_environment(env)

        # Check for cross-agent insights
        cross = self.memory.get_cross_agent_insights(self.agent_id)
        if cross:
            suggestion["cross_agent_insights"] = [
                {"from": c["source_agent"], "insight": c["description"]}
                for c in cross[:3]
            ]

        suggestion["environment"] = env_analysis
        suggestion["cycle"] = self.cycle
        suggestion["best_metric"] = self.best_metric
        suggestion["experiments_since_improvement"] = self.experiments_since_improvement

        # Log suggestion
        self._log(f"CYCLE {self.cycle} SUGGESTION", suggestion)
        return suggestion

    def observe(self, action: dict, result: dict) -> Observation:
        """
        Create a full 12D observation of what just happened.

        action: dict with keys like {type, description, diff}
        result: dict with keys like {metric_name, metric_value, outcome, ...}
        """
        metric_value = result.get("metric_value", 0.0)
        prev_metric = self._history[-1].metric_value if self._history else metric_value
        metric_delta = prev_metric - metric_value  # positive = improvement (lower is better)

        outcome = result.get("outcome", "discard")
        if outcome == "keep":
            self.experiments_since_improvement = 0
            if metric_value < self.best_metric:
                self.best_metric = metric_value
                self.best_metric_cycle = self.cycle
        else:
            self.experiments_since_improvement += 1

        # Score 12 dimensions (Phase C: raw observation)
        scores = self._compute_12d_scores(action, result, metric_delta)

        # Phase B: 4-Element Deep Synthesis
        # Run through all 4 lenses and compute element-weighted scores
        lens_results = {}
        for lens_name in ELEMENT_LENSES:
            lens_results[lens_name] = apply_lens(scores, lens_name)

        # Propagate cross-dimensional coupling
        scores = propagate_coupling(scores)

        # Compute differential calculus (using Riemannian metric)
        velocity, acceleration, jerk = self._compute_calculus(scores)

        # Compute witness hash
        obs_payload = f"{self._prev_hash}:{self.cycle}:{metric_value}:{outcome}"
        obs_hash = hashlib.sha256(obs_payload.encode()).hexdigest()[:16]

        # Match against known patterns
        pattern_match = self._match_pattern(action.get("type", ""))

        obs = Observation(
            cycle_id=self.cycle,
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent_id=self.agent_id,
            project=self.project,
            action_type=action.get("type", ""),
            action_description=action.get("description", ""),
            action_diff=action.get("diff", "")[:2000],  # truncate large diffs
            metric_name=result.get("metric_name", ""),
            metric_value=metric_value,
            metric_delta=metric_delta,
            outcome=outcome,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            tags=json.dumps(action.get("tags", [])),
            pattern_match=pattern_match,
            strategy_used=action.get("strategy", ""),
            environment_state=json.dumps(result.get("environment", {})),
            prev_hash=self._prev_hash,
            obs_hash=obs_hash,
            **scores,
        )

        # Store
        self.memory.store_observation(obs)
        self._history.append(obs)
        self._prev_hash = obs_hash

        # Update agent stats
        self.memory.update_agent_stats(self.agent_id, self.cycle, self.best_metric)

        # Detect contradictions (Phase E: observation ledger)
        self._detect_contradictions(action, result, scores, metric_delta)

        # Detect emergence events (Phase E: observation ledger)
        self._detect_emergence(action, result, scores, metric_delta, lens_results)

        # Record strategy outcome for meta-learning
        strategy_used = action.get("strategy", "unknown")
        self.memory.record_strategy_outcome(
            self.agent_id, self.cycle, strategy_used, outcome, metric_delta
        )
        self.strategy.record_strategy_outcome(strategy_used, outcome, metric_delta)

        # Periodic pattern extraction (every 10 cycles)
        if self.cycle % 10 == 0:
            patterns = self.memory.extract_patterns()
            self._log(f"CYCLE {self.cycle} PATTERNS", [asdict(p) for p in patterns])
            self._share_insights_to_other_agents(patterns)

        # Log observation
        self._log(f"CYCLE {self.cycle} OBSERVATION", {
            "outcome": outcome,
            "metric": metric_value,
            "delta": metric_delta,
            "velocity": velocity,
            "acceleration": acceleration,
            "pattern_match": pattern_match,
            "12d_magnitude": obs.magnitude(),
        })

        return obs

    def decide(self, result: dict) -> str:
        """
        Make the keep/discard decision and log it.
        Returns "keep", "discard", or "crash".
        """
        outcome = result.get("outcome", "discard")

        if outcome == "keep":
            self._log(f"CYCLE {self.cycle} DECISION", {
                "decision": "KEEP",
                "metric": result.get("metric_value"),
                "improvement": result.get("metric_delta", 0),
                "total_kept": sum(1 for o in self._history if o.outcome == "keep"),
            })
        elif outcome == "crash":
            self._log(f"CYCLE {self.cycle} DECISION", {
                "decision": "CRASH",
                "error": result.get("error", "unknown"),
            })
        else:
            self._log(f"CYCLE {self.cycle} DECISION", {
                "decision": "DISCARD",
                "metric": result.get("metric_value"),
                "regression": result.get("metric_delta", 0),
            })

        return outcome

    def report(self) -> dict:
        """
        Generate a comprehensive report of all learning to date.
        Includes patterns, strategies, cross-agent insights, and recommendations.
        """
        patterns = self.memory.extract_patterns()
        stats = self.memory.get_success_rate_by_type()
        other_agents = self.memory.get_other_agents(self.agent_id)
        recent = self.memory.get_recent_observations(self.agent_id, limit=50)

        # Compute overall trajectory
        if len(self._history) >= 2:
            first = self._history[0].metric_value
            last = self._history[-1].metric_value
            total_improvement = first - last
        else:
            total_improvement = 0

        # Get contradiction and emergence data
        open_contradictions = self.memory.get_open_contradictions(self.agent_id)
        emergence_events = self.memory.get_emergence_events(self.agent_id)
        strategy_effectiveness = self.memory.get_strategy_effectiveness(self.agent_id)

        report = {
            "agent_id": self.agent_id,
            "project": self.project,
            "total_cycles": self.cycle,
            "best_metric": self.best_metric,
            "best_metric_cycle": self.best_metric_cycle,
            "total_improvement": total_improvement,
            "current_strategy": self.strategy._determine_strategy(self.cycle, recent),
            "experiments_since_improvement": self.experiments_since_improvement,
            "patterns": [asdict(p) for p in patterns],
            "success_rates": stats,
            "other_agents": other_agents,
            "top_recommendations": self._generate_recommendations(patterns),
            # NEW: Self-improvement data
            "open_contradictions": open_contradictions,
            "emergence_events": [dict(e) for e in emergence_events],
            "strategy_effectiveness": strategy_effectiveness,
            "4_element_summary": {
                lens: ELEMENT_LENSES[lens]["focus"]
                for lens in ELEMENT_LENSES
            },
        }

        self._log("REPORT", report)
        return report

    def _detect_contradictions(self, action: dict, result: dict,
                                scores: dict, metric_delta: float):
        """
        Detect contradictions — when expectations don't match outcomes,
        or when improving one dimension degrades another.
        """
        # Contradiction 1: Expected improvement, got regression
        expected = action.get("expected_outcome", "unknown")
        outcome = result.get("outcome", "discard")
        if expected == "improve" and outcome == "discard":
            self.memory.record_contradiction(
                self.agent_id, self.cycle,
                f"Expected improvement from {action.get('type', '?')} but got regression. "
                f"Delta: {metric_delta:.6f}",
                "expectation", "reality", severity=0.7
            )

        # Contradiction 2: High structure score but low legibility (coupling violation)
        if scores.get("x1_structure", 0) > 0.7 and scores.get("x7_legibility", 0) < 0.3:
            self.memory.record_contradiction(
                self.agent_id, self.cycle,
                "High structure change but low legibility — complex change not readable",
                "x1_structure", "x7_legibility", severity=0.5
            )

        # Contradiction 3: High grounding but low coordination
        if scores.get("x9_grounding", 0) > 0.8 and scores.get("x3_coordination", 0) < 0.3:
            self.memory.record_contradiction(
                self.agent_id, self.cycle,
                "Strong metric evidence but poor coordination — siloed improvement",
                "x9_grounding", "x3_coordination", severity=0.4
            )

    def _detect_emergence(self, action: dict, result: dict,
                           scores: dict, metric_delta: float,
                           lens_results: dict):
        """
        Detect emergence events — unexpected coherent improvements
        that increase scores across multiple dimensions.
        """
        if result.get("outcome") != "keep" or metric_delta <= 0:
            return

        # Count how many dimensions are above 0.6 (multi-dimensional coherence)
        high_dims = [k for k, v in scores.items()
                     if k.startswith("x") and v > 0.6]

        # Check novelty: is this action type unusual?
        stats = self.memory.get_success_rate_by_type()
        action_type = action.get("type", "unknown")
        type_stats = stats.get(action_type, {})
        novelty = 1.0 - type_stats.get("success_rate", 0.5)  # rare success = high novelty

        # Multi-dimensional coherence gain
        coherence = len(high_dims) / 12.0

        # Reuse potential: can this be reproduced?
        reuse = 0.8 if action.get("diff", "") else 0.3

        # Emergence criteria: novel + coherent + significant
        if len(high_dims) >= 3 and novelty > 0.3 and metric_delta > 0:
            self.memory.record_emergence(
                self.agent_id, self.cycle,
                f"Emergence via {action_type}: {len(high_dims)} dimensions above 0.6, "
                f"novelty={novelty:.2f}, delta={metric_delta:.6f}",
                high_dims, coherence, novelty, reuse,
            )
            self._log(f"CYCLE {self.cycle} EMERGENCE DETECTED", {
                "action_type": action_type,
                "dimensions_above_0.6": high_dims,
                "coherence": coherence,
                "novelty": novelty,
                "metric_delta": metric_delta,
            })

    def _compute_12d_scores(self, action: dict, result: dict,
                            metric_delta: float) -> dict:
        """
        Compute 12D observation scores based on action and result.
        These are heuristic scores that improve as more data accumulates.
        """
        outcome = result.get("outcome", "discard")
        is_success = outcome == "keep"

        # Base scores
        scores = {}

        # x1: Structure — did the action change architecture?
        action_type = action.get("type", "")
        scores["x1_structure"] = 0.8 if "architecture" in action_type else 0.4

        # x2: Semantics — is the action well-described?
        desc_len = len(action.get("description", ""))
        scores["x2_semantics"] = min(1.0, desc_len / 100)

        # x3: Coordination — are we using cross-agent insights?
        scores["x3_coordination"] = 0.8 if action.get("from_cross_agent") else 0.3

        # x4: Recursion — are we building on prior learning?
        scores["x4_recursion"] = 0.9 if action.get("strategy") in ("exploit", "combine") else 0.3

        # x5: Contradiction — did this contradict expectations?
        pattern = action.get("expected_outcome", "unknown")
        if pattern == "improve" and not is_success:
            scores["x5_contradiction"] = 0.2  # Low = contradiction detected
        elif pattern == "degrade" and is_success:
            scores["x5_contradiction"] = 0.3  # Unexpected success — interesting!
        else:
            scores["x5_contradiction"] = 0.7

        # x6: Emergence — unexpected positive result?
        if is_success and metric_delta > 0:
            scores["x6_emergence"] = min(1.0, 0.5 + abs(metric_delta) * 100)
        else:
            scores["x6_emergence"] = 0.3

        # x7: Legibility — is the diff readable?
        diff_len = len(action.get("diff", ""))
        scores["x7_legibility"] = max(0.2, 1.0 - diff_len / 5000)

        # x8: Routing — was the right strategy used?
        scores["x8_routing"] = 0.8 if action.get("strategy") else 0.4

        # x9: Grounding — do we have solid metrics?
        has_metric = result.get("metric_value", 0) > 0
        scores["x9_grounding"] = 0.9 if has_metric else 0.2

        # x10: Compression — small change, big effect?
        if is_success and diff_len > 0:
            efficiency = abs(metric_delta) / max(diff_len, 1) * 10000
            scores["x10_compression"] = min(1.0, efficiency)
        else:
            scores["x10_compression"] = 0.3

        # x11: Interop — does this help other agents?
        scores["x11_interop"] = 0.7 if is_success else 0.4

        # x12: Potential — does this open new directions?
        scores["x12_potential"] = 0.8 if is_success and scores["x6_emergence"] > 0.6 else 0.4

        return scores

    def _compute_calculus(self, scores: dict) -> tuple[float, float, float]:
        """Compute velocity, acceleration, jerk from observation history."""
        current = sum(scores.values()) / len(scores)

        if len(self._history) >= 1:
            prev = self._history[-1].magnitude() / math.sqrt(12)
            velocity = current - prev
        else:
            velocity = 0.0

        if len(self._history) >= 2:
            prev_prev = self._history[-2].magnitude() / math.sqrt(12)
            prev = self._history[-1].magnitude() / math.sqrt(12)
            prev_velocity = prev - prev_prev
            acceleration = velocity - prev_velocity
        else:
            acceleration = 0.0

        if len(self._history) >= 3:
            pp = self._history[-3].magnitude() / math.sqrt(12)
            p = self._history[-2].magnitude() / math.sqrt(12)
            c = self._history[-1].magnitude() / math.sqrt(12)
            prev_acc = (c - p) - (p - pp)
            jerk = acceleration - prev_acc
        else:
            jerk = 0.0

        return velocity, acceleration, jerk

    def _match_pattern(self, action_type: str) -> str:
        """Check if this action type matches a known pattern."""
        patterns = self.memory.extract_patterns(min_samples=2)
        for p in patterns:
            if p.action_type == action_type:
                return f"{p.pattern_id}: {p.success_rate:.0%} success ({p.sample_count} samples)"
        return ""

    def _share_insights_to_other_agents(self, patterns: list[Pattern]):
        """Share successful patterns with other registered agents."""
        other_agents = self.memory.get_other_agents(self.agent_id)
        for agent in other_agents:
            for p in patterns:
                if p.success_rate > 0.5 and p.confidence > 0.3 and p.sample_count >= 5:
                    self.memory.share_insight(
                        source=self.agent_id,
                        target=agent["agent_id"],
                        insight_type="successful_strategy",
                        description=f"{p.action_type}: {p.success_rate:.0%} success rate, "
                                   f"avg improvement {p.avg_improvement:.6f}",
                        evidence=json.dumps(asdict(p)),
                    )

    def _generate_recommendations(self, patterns: list[Pattern]) -> list[dict]:
        """Generate top recommendations based on all accumulated learning."""
        recs = []

        # Top performing patterns
        for p in sorted(patterns, key=lambda x: x.success_rate * x.confidence, reverse=True)[:3]:
            recs.append({
                "type": "proven_strategy",
                "action": p.action_type,
                "reason": f"{p.success_rate:.0%} success across {p.sample_count} experiments",
                "confidence": p.confidence,
            })

        # Under-explored areas with potential
        for p in patterns:
            ctx = json.loads(p.context_conditions) if p.context_conditions else {}
            if ctx.get("trend") == "improving":
                recs.append({
                    "type": "improving_trend",
                    "action": p.action_type,
                    "reason": f"Success rate improving over time (late: {ctx.get('late_success_rate', '?')})",
                    "confidence": p.confidence * 0.8,
                })

        return recs

    def _snapshot_environment(self) -> EnvironmentSnapshot:
        """Take a snapshot of the current environment."""
        env = EnvironmentSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            current_best_metric=self.best_metric,
            total_experiments=self.cycle,
            experiments_since_improvement=self.experiments_since_improvement,
            active_agents=len(self.memory.get_other_agents(self.agent_id)) + 1,
        )

        # Try to get GPU info (if available)
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                if len(parts) >= 3:
                    env.gpu_util = float(parts[0].strip())
                    env.gpu_memory_used_mb = float(parts[1].strip())
                    env.gpu_memory_total_mb = float(parts[2].strip())
        except Exception:
            pass

        return env

    def _log(self, label: str, data: Any):
        """Write to the observation log."""
        log_file = self.log_dir / f"{self.agent_id}.jsonl"
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "data": data,
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.memory.close()
