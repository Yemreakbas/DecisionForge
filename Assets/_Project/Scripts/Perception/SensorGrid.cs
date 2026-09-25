using System.Collections.Generic;
using UnityEngine;

namespace JevNpcBrain.Perception
{
    using Arena;

    /// <summary>
    /// Fills a <see cref="TacticalObservation"/> for one agent.
    ///
    /// This is the only place allowed to look at the world directly. It simulates
    /// eyes and ears; what comes out the other side is the single input both
    /// brains share. If a brain ever needs something this class does not provide,
    /// the fix is to add a channel here -- never to reach past it.
    ///
    /// Cost note: visibility and threat exposure are the expensive channels, since
    /// both walk a line per cell. We cap threat exposure to the two strongest
    /// belief tracks, which keeps a 10 Hz tick affordable for six agents. If the
    /// profiler complains in phase 5, this is the first thing to optimise.
    /// </summary>
    public sealed class SensorGrid
    {
        /// <summary>Beyond this the agent cannot see, regardless of geometry.</summary>
        public float SightRange = 40f;

        /// <summary>Half-angle of the vision cone, degrees.</summary>
        public float SightHalfAngle = 80f;

        public float MemoryHorizon = 8f;

        private readonly List<NoiseEvents.Noise> _noises = new List<NoiseEvents.Noise>(16);
        private readonly List<EnemyBelief.Track> _threats = new List<EnemyBelief.Track>(4);

        public void Populate(
            TacticalObservation obs,
            Vector3 selfPosition,
            Vector3 selfForward,
            EnemyBelief belief,
            IReadOnlyList<Vector3> allyPositions,
            int team,
            float time)
        {
            var layout = ArenaLayout.Current;
            if (layout == null) return;

            obs.Clear();
            obs.Origin = selfPosition;
            obs.Timestamp = time;

            SelectThreats(belief);

            WriteObstacles(obs, layout);
            WriteBelief(obs, belief);
            WriteAllies(obs, allyPositions);
            WriteNoise(obs, team, time);
            WriteVisibility(obs, layout, selfPosition, selfForward);
            WriteThreatExposure(obs, layout);
            WriteCoverQuality(obs, layout);

            BuildSummary(obs, layout, selfPosition, belief, allyPositions);
        }

        /// <summary>Two strongest tracks. More than that is noise, not information.</summary>
        private void SelectThreats(EnemyBelief belief)
        {
            _threats.Clear();
            var tracks = belief.Tracks;

            for (int i = 0; i < tracks.Count; i++)
            {
                var t = tracks[i];
                if (t.Confidence < 0.1f) continue;

                if (_threats.Count < 2) { _threats.Add(t); continue; }

                int weakest = _threats[0].Confidence <= _threats[1].Confidence ? 0 : 1;
                if (t.Confidence > _threats[weakest].Confidence) _threats[weakest] = t;
            }
        }

        private static void WriteObstacles(TacticalObservation obs, ArenaLayout layout)
        {
            for (int y = 0; y < TacticalObservation.GridSize; y++)
            {
                for (int x = 0; x < TacticalObservation.GridSize; x++)
                {
                    var world = obs.CellToWorld(x, y);
                    obs.Set(ObservationChannel.ObstacleHeight, x, y, layout.SampleHeight(world));
                }
            }
        }

        /// <summary>
        /// Splats each track as a blob whose radius is its own uncertainty. A track
        /// we just saw is a point; one we lost ten seconds ago is a smear, and the
        /// brain can tell the difference by looking at the shape.
        /// </summary>
        private static void WriteBelief(TacticalObservation obs, EnemyBelief belief)
        {
            var tracks = belief.Tracks;

            for (int i = 0; i < tracks.Count; i++)
            {
                var t = tracks[i];
                if (t.Confidence < 0.02f) continue;
                Splat(obs, ObservationChannel.EnemyBelief, t.Position, t.Spread + 1f, t.Confidence);
            }
        }

        private static void WriteAllies(TacticalObservation obs, IReadOnlyList<Vector3> allies)
        {
            if (allies == null) return;
            for (int i = 0; i < allies.Count; i++)
                Splat(obs, ObservationChannel.AllyPresence, allies[i], 1.5f, 1f);
        }

        private void WriteNoise(TacticalObservation obs, int team, float time)
        {
            NoiseEvents.Collect(team, time, _noises);

            for (int i = 0; i < _noises.Count; i++)
            {
                var n = _noises[i];
                float freshness = 1f - Mathf.Clamp01((time - n.Time) / NoiseEvents.Lifetime);
                if (freshness <= 0f) continue;
                Splat(obs, ObservationChannel.NoiseHeat, n.Position, 2.5f, n.Loudness * freshness);
            }
        }

        /// <summary>Gaussian-ish blob, clamped to the window.</summary>
        private static void Splat(TacticalObservation obs, ObservationChannel channel,
            Vector3 world, float radius, float weight)
        {
            if (!obs.TryWorldToCell(world, out int cx, out int cy)) return;

            int cells = Mathf.Max(1, Mathf.CeilToInt(radius / TacticalObservation.CellSize));
            float inv = 1f / Mathf.Max(radius, 0.01f);

            for (int y = cy - cells; y <= cy + cells; y++)
            {
                for (int x = cx - cells; x <= cx + cells; x++)
                {
                    if (!TacticalObservation.InBounds(x, y)) continue;

                    float d = Vector3.Distance(obs.CellToWorld(x, y), world) * inv;
                    if (d > 1f) continue;

                    obs.Accumulate(channel, x, y, weight * (1f - d * d));
                }
            }
        }

        private void WriteVisibility(TacticalObservation obs, ArenaLayout layout,
            Vector3 eye, Vector3 forward)
        {
            float cos = Mathf.Cos(SightHalfAngle * Mathf.Deg2Rad);
            var flatForward = new Vector3(forward.x, 0f, forward.z).normalized;

            for (int y = 0; y < TacticalObservation.GridSize; y++)
            {
                for (int x = 0; x < TacticalObservation.GridSize; x++)
                {
                    if (obs.Get(ObservationChannel.ObstacleHeight, x, y) >= 1f) continue;

                    var world = obs.CellToWorld(x, y);
                    var delta = world - eye;
                    delta.y = 0f;

                    float distance = delta.magnitude;
                    if (distance > SightRange) continue;
                    if (distance > 0.5f && Vector3.Dot(delta / distance, flatForward) < cos) continue;
                    if (!layout.HasLineOfSight(eye, world)) continue;

                    obs.Set(ObservationChannel.Visibility, x, y, 1f);
                }
            }
        }

        /// <summary>
        /// How dangerous each cell is: the weighted count of believed enemy
        /// positions that can see it. This is the channel a good decision is made
        /// of -- "where can I stand that they cannot shoot" -- and it is derived
        /// purely from belief, so an agent that has lost track of someone is
        /// genuinely, correctly uncertain about where the danger is.
        /// </summary>
        private void WriteThreatExposure(TacticalObservation obs, ArenaLayout layout)
        {
            if (_threats.Count == 0) return;

            for (int t = 0; t < _threats.Count; t++)
            {
                var threat = _threats[t];

                for (int y = 0; y < TacticalObservation.GridSize; y++)
                {
                    for (int x = 0; x < TacticalObservation.GridSize; x++)
                    {
                        if (obs.Get(ObservationChannel.ObstacleHeight, x, y) >= 1f) continue;

                        var world = obs.CellToWorld(x, y);
                        if (Vector3.Distance(world, threat.Position) > 45f) continue;
                        if (!layout.HasLineOfSight(threat.Position, world)) continue;

                        obs.Accumulate(ObservationChannel.ThreatExposure, x, y, threat.Confidence);
                    }
                }
            }

            float inv = 1f / _threats.Count;
            for (int y = 0; y < TacticalObservation.GridSize; y++)
                for (int x = 0; x < TacticalObservation.GridSize; x++)
                    obs.Set(ObservationChannel.ThreatExposure, x, y,
                        Mathf.Clamp01(obs.Get(ObservationChannel.ThreatExposure, x, y) * inv));
        }

        /// <summary>
        /// Cover is the complement of exposure, but only where something is
        /// actually next to you. A cell in the open that happens to be out of
        /// sight is lucky, not covered, and the difference shows up the moment the
        /// enemy moves.
        /// </summary>
        private static void WriteCoverQuality(TacticalObservation obs, ArenaLayout layout)
        {
            for (int y = 0; y < TacticalObservation.GridSize; y++)
            {
                for (int x = 0; x < TacticalObservation.GridSize; x++)
                {
                    if (obs.Get(ObservationChannel.ObstacleHeight, x, y) >= 1f) continue;

                    float adjacent = 0f;
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            if (dx == 0 && dy == 0) continue;
                            if (!TacticalObservation.InBounds(x + dx, y + dy)) continue;
                            adjacent = Mathf.Max(adjacent,
                                obs.Get(ObservationChannel.ObstacleHeight, x + dx, y + dy));
                        }
                    }

                    if (adjacent <= 0f) continue;

                    float exposure = obs.Get(ObservationChannel.ThreatExposure, x, y);
                    obs.Set(ObservationChannel.CoverQuality, x, y,
                        Mathf.Clamp01(adjacent * (1f - exposure)));
                }
            }
        }

        /// <summary>
        /// The scalar digest the baseline reads. Computed from the same channels
        /// the learned brain gets, so this is a convenience, never extra evidence.
        /// </summary>
        private void BuildSummary(TacticalObservation obs, ArenaLayout layout,
            Vector3 self, EnemyBelief belief, IReadOnlyList<Vector3> allies)
        {
            const int c = TacticalObservation.CenterCell;
            var s = new ObservationSummary
            {
                CoverQualityHere = obs.Get(ObservationChannel.CoverQuality, c, c),
                ThreatExposureHere = obs.Get(ObservationChannel.ThreatExposure, c, c),
                EnemyBeliefMass = Mathf.Clamp01(belief.TotalMass()),
                DistanceToCenter = Vector3.Distance(self, layout.CenterPosition)
            };

            var threatAxis = Vector3.zero;
            if (belief.TryGetStrongest(out var strongest))
            {
                s.EnemyBeliefCentroid = strongest.Position;
                s.EnemyBeliefDistance = Vector3.Distance(self, strongest.Position);
                s.EnemyBeliefSpread = Mathf.Clamp01(strongest.Spread / 9f);
                threatAxis = (strongest.Position - self);
                threatAxis.y = 0f;
                threatAxis.Normalize();
            }
            else
            {
                s.EnemyBeliefDistance = 999f;
                s.EnemyBeliefSpread = 1f;
            }

            ScoreCoverAnchors(ref s, layout, self, threatAxis);

            s.NearestAllyDistance = 999f;
            if (allies != null)
            {
                for (int i = 0; i < allies.Count; i++)
                {
                    float d = Vector3.Distance(self, allies[i]);
                    if (d >= s.NearestAllyDistance) continue;
                    s.NearestAllyDistance = d;
                    s.HasAlly = true;
                }
            }

            // How contested the centre looks right now, straight off the belief
            // channel rather than from ground truth.
            if (obs.TryWorldToCell(layout.CenterPosition, out int cx, out int cy))
            {
                float pressure = 0f;
                int samples = 0;
                int span = Mathf.CeilToInt(layout.CenterRadius / TacticalObservation.CellSize);

                for (int y = cy - span; y <= cy + span; y++)
                {
                    for (int x = cx - span; x <= cx + span; x++)
                    {
                        if (!TacticalObservation.InBounds(x, y)) continue;
                        pressure += obs.Get(ObservationChannel.EnemyBelief, x, y);
                        samples++;
                    }
                }

                s.CenterPressure = samples > 0 ? Mathf.Clamp01(pressure / samples) : 0f;
            }

            obs.Summary = s;
        }

        /// <summary>
        /// Picks the best cover to move to, and separately the best one that is NOT
        /// on the threat axis. Keeping those two apart is what lets the baseline
        /// distinguish "get behind something" from "come at them from the side" --
        /// without it, every advance collapses into the same frontal push.
        /// </summary>
        private static void ScoreCoverAnchors(ref ObservationSummary s, ArenaLayout layout,
            Vector3 self, Vector3 threatAxis)
        {
            var anchors = layout.CoverAnchors;
            float bestScore = float.NegativeInfinity;
            float bestFlankScore = float.NegativeInfinity;

            bool hasThreat = threatAxis.sqrMagnitude > 0.01f;

            for (int i = 0; i < anchors.Count; i++)
            {
                var anchor = anchors[i];
                float distance = Vector3.Distance(self, anchor.Position);
                if (distance > 24f) continue;

                // A cover only counts if it sits between us and the threat.
                float facing = hasThreat ? Mathf.Clamp01(Vector3.Dot(anchor.Normal, threatAxis)) : 0.5f;
                float quality = anchor.Height * (0.35f + 0.65f * facing);
                float score = quality - distance / 30f;

                if (score > bestScore)
                {
                    bestScore = score;
                    s.HasBestCover = true;
                    s.BestCoverQuality = Mathf.Clamp01(quality);
                    s.BestCoverDistance = distance;
                    s.BestCoverPosition = anchor.Position;
                }

                if (!hasThreat) continue;

                var toAnchor = anchor.Position - self;
                toAnchor.y = 0f;
                if (toAnchor.sqrMagnitude < 0.01f) continue;

                // Off-axis means we approach from a direction they are not watching.
                float offAxis = 1f - Mathf.Abs(Vector3.Dot(toAnchor.normalized, threatAxis));
                if (offAxis < 0.35f) continue;

                float flankScore = quality * offAxis - distance / 40f;
                if (flankScore <= bestFlankScore) continue;

                bestFlankScore = flankScore;
                s.HasFlankCover = true;
                s.FlankCoverQuality = Mathf.Clamp01(quality * offAxis);
                s.FlankCoverPosition = anchor.Position;
            }
        }
    }
}
